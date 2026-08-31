"""Run the QA intent registry over a deterministic corpus sample.

Emits:
  - <out-dir>/<triples-name>            (jsonl triples)
  - <out-dir>/examples/group_<G>.jsonl  (20 random triples/group + code)
  - <report>                            (COVERAGE_REPORT.md)

Sampling is deterministic: seeded shuffle of split indices (repo-ordered
corpus makes first-N samples degenerate). Stacks (group I) partition the
parseable sample rows by md5(qualified) order into stacks of 8 with unique
function names per stack.

Usage:
  uv run python dataset/qa_coverage.py --limit 5000  --triples-name triples_dev.jsonl
  uv run python dataset/qa_coverage.py --limit 100000 --triples-name triples_sample.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore", category=SyntaxWarning)

import qa_exec
import qa_extract
from qa_extract import (REGISTRY, ParseFailure, extract_row, extract_stack_intents,
                        fact_to_triple, leak_check, make_ctx)

SAMPLE_SEED = 20260806
STACK_SIZE = 8
GROUP_QUOTAS = {"A": 20, "B": 15, "C": 15, "D": 15, "E/F": 12, "G": 5, "H": 8, "I": 10}

_POOLS = None  # worker-global distractor pools


def _init_worker(pools):
    global _POOLS
    _POOLS = pools
    warnings.filterwarnings("ignore", category=SyntaxWarning)


# --------------------------------------------------------------------------
# Phase A: distractor pools (docstring first lines for C5)
# --------------------------------------------------------------------------


def _pool_worker(chunk):
    out = []
    for row in chunk:
        try:
            ctx = make_ctx(row, -1)
        except Exception:
            continue
        first = qa_extract._first_docstring_line(ctx)
        if first and 15 <= len(first) <= 160:
            out.append(first)
    return out


def build_pools(rows, workers):
    doc_rows = [r for r in rows if r["has_docstring"]][:30000]
    chunks = [doc_rows[i:i + 500] for i in range(0, len(doc_rows), 500)]
    pool_lines = []
    with mp.Pool(workers) as p:
        for res in p.imap_unordered(_pool_worker, chunks):
            pool_lines.extend(res)
    # deterministic order regardless of imap scheduling
    pool_lines = sorted(set(pool_lines))
    return {"doc_first_lines": pool_lines}


# --------------------------------------------------------------------------
# Phase B: per-function extraction (+ screen + exec-task synthesis)
# --------------------------------------------------------------------------


def _extract_worker(chunk):
    """chunk: [(row_idx, row)]. Returns (triples, stats, exec_tasks, ok_idxs)."""
    triples, exec_tasks, ok = [], [], []
    stats = {
        "rows": 0, "parse_fail": 0, "normalized": 0, "leak_dropped": 0,
        "applicable": Counter(), "emitted": Counter(),
        "screen_pass": 0, "screen_reasons": Counter(),
    }
    for row_idx, row in chunk:
        stats["rows"] += 1
        try:
            ctx = make_ctx(row, row_idx, pools=_POOLS)
        except (ParseFailure, SyntaxError, RecursionError, ValueError, MemoryError):
            stats["parse_fail"] += 1
            continue
        ok.append(row_idx)
        if ctx.normalized:
            stats["normalized"] += 1
        s_ok, s_reason, _ = qa_exec.screen(ctx.fn, row)
        ctx.self_contained = s_ok
        if s_ok:
            stats["screen_pass"] += 1
            try:
                combos, _ = qa_exec.synth_inputs(ctx.fn, row["qualified"])
                exec_tasks.append({
                    "qualified": row["qualified"], "row_idx": row_idx,
                    "code": ctx.norm_code, "fn_name": ctx.fn.name,
                    "combos": combos,
                })
            except Exception:
                pass
        else:
            stats["screen_reasons"][s_reason.split(":")[0]] += 1
        facts, fstats = extract_row(ctx)
        stats["leak_dropped"] += fstats["leak_dropped"]
        for iid in fstats["applicable"]:
            stats["applicable"][iid] += 1
        for f in facts:
            stats["emitted"][f.intent] += 1
            triples.append(fact_to_triple(f, row, row_idx))
    return triples, stats, exec_tasks, ok


# --------------------------------------------------------------------------
# Stacks (group I)
# --------------------------------------------------------------------------


def build_stacks(rows_by_idx, ok_idxs):
    """Deterministic stacks of STACK_SIZE with unique short names."""
    order = sorted(ok_idxs, key=lambda i: hashlib.md5(
        rows_by_idx[i]["qualified"].encode()).hexdigest())
    stacks, cur, cur_names, overflow = [], [], set(), []
    for i in order:
        name = rows_by_idx[i]["func_name"].split(".")[-1]
        if name in cur_names:
            overflow.append(i)
            continue
        cur.append(i)
        cur_names.add(name)
        if len(cur) == STACK_SIZE:
            stacks.append(cur)
            cur, cur_names = [], set()
    # one overflow pass
    for i in overflow:
        name = rows_by_idx[i]["func_name"].split(".")[-1]
        if name not in cur_names:
            cur.append(i)
            cur_names.add(name)
            if len(cur) == STACK_SIZE:
                stacks.append(cur)
                cur, cur_names = [], set()
    return stacks


def extract_stacks(stacks, rows_by_idx, stats):
    triples = []
    n = len(stacks)
    for si, idxs in enumerate(stacks):
        rows = []
        for i in idxs:
            r = dict(rows_by_idx[i])
            r["row_idx"] = i
            rows.append(r)
        stack = {"rows": rows, "row_idxs": idxs}
        # decoy: target-ish row from the next stack (deterministic)
        decoy_idx = stacks[(si + 1) % n][0] if n > 1 else None
        decoy = None
        if decoy_idx is not None:
            decoy = dict(rows_by_idx[decoy_idx])
            decoy["row_idx"] = decoy_idx
        try:
            pairs = extract_stack_intents(stack, decoy)
        except Exception:
            continue
        for row, fact in pairs:
            stats["applicable"][fact.intent] += 1
            if not leak_check(fact):
                stats["leak_dropped"] += 1
                continue
            stats["emitted"][fact.intent] += 1
            triples.append(fact_to_triple(fact, row, row["row_idx"]))
    return triples


# --------------------------------------------------------------------------
# Exec stage
# --------------------------------------------------------------------------


def run_exec_stage(exec_tasks, workers, stats):
    triples = []
    # 484 fork near-dups share a qualified name with different code; the child
    # echoes only qualified, so a name-keyed join would cross-attribute answers.
    # Exclude colliding names outright (matches the pod artifact's scrub).
    from collections import Counter
    _qc = Counter(t["qualified"] for t in exec_tasks)
    _collided = {q for q, n in _qc.items() if n > 1}
    if _collided:
        stats["exec_status"]["dropped_name_collision"] = sum(
            _qc[q] for q in _collided)
    exec_tasks = [t for t in exec_tasks if t["qualified"] not in _collided]
    task_by_q = {t["qualified"]: t for t in exec_tasks}
    batches = [exec_tasks[i:i + 40] for i in range(0, len(exec_tasks), 40)]
    records = []
    with ThreadPoolExecutor(max_workers=workers) as tp:
        for recs in tp.map(qa_exec.run_sandbox_batch, batches):
            records.extend(recs)
    exec_ok_rows = set()
    for rec in records:
        stats["exec_status"][rec["status"]] += 1
        task = task_by_q.get(rec["qualified"])
        if task is None or rec["status"] != "ok":
            continue
        for run in rec.get("runs", []):
            stats["run_status"][run.get("status", "?")] += 1
        try:
            tree = ast.parse(task["code"])
            fn = next(n for n in tree.body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        except Exception:
            continue
        facts = qa_exec.build_exec_facts(rec, fn, rec["qualified"])
        row = {"qualified": rec["qualified"]}
        for f in facts:
            stats["applicable"][f.intent] += 1
            if not leak_check(f):
                stats["leak_dropped"] += 1
                continue
            stats["emitted"][f.intent] += 1
            if f.intent.startswith("D"):
                exec_ok_rows.add(task["row_idx"])
            triples.append(fact_to_triple(f, row, task["row_idx"]))
    stats["exec_rows_with_facts"] = len(exec_ok_rows)
    return triples


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

SPEC_NOTES = {
    "C3": "purpose distractor-MCQ needs generated purpose texts -> LUNA slot only in this build",
    "C5": "no LLM: implemented as description-match MCQ (own docstring 1st line vs 3 pool docstrings)",
    "H2": "injected-mismatch variant (docstring mutation) deferred to phrasing stage; natural match/mismatch emitted",
}

DISTRACTOR_RECIPES = {
    "C5": "3 docstring first lines drawn rng(md5(qualified)) from the corpus-wide pool (sorted, deduped)",
    "C7": "3 roles rng-drawn from the fixed 7-role taxonomy minus the correct role",
    "H3": "the 3 other docstring styles from the fixed 4-style set",
    "I3": "3 sibling function names from the same stack",
}


def write_report(path, args, stats, triples_all, n_rows, timings, n_stacks, pools):
    ap, em = stats["applicable"], stats["emitted"]
    by_group = defaultdict(lambda: [0, 0])
    intents_sorted = sorted(REGISTRY, key=lambda i: (REGISTRY[i].group, i))
    for iid in intents_sorted:
        by_group[REGISTRY[iid].group][0] += ap.get(iid, 0)
        by_group[REGISTRY[iid].group][1] += em.get(iid, 0)
    total_em = sum(em.values()) or 1
    parse_ok = n_rows - stats["parse_fail"]

    L = []
    L.append("# QA extractor — coverage report")
    L.append("")
    L.append(f"Generated by dataset/qa_coverage.py — split={args.split}, "
             f"sample={n_rows} rows (seeded shuffle, seed {SAMPLE_SEED}), "
             f"workers={args.workers}.")
    L.append("")
    L.append("## Headline")
    L.append(f"- rows: {n_rows} | parse ok: {parse_ok} "
             f"({100*parse_ok/n_rows:.1f}%) | indent-normalized: "
             f"{stats['normalized']} | parse failures: {stats['parse_fail']}")
    L.append(f"- triples emitted: {sum(em.values())} "
             f"({sum(em.values())/max(parse_ok,1):.2f} per parseable fn)")
    L.append(f"- self-contained (exec screen pass): {stats['screen_pass']} "
             f"({100*stats['screen_pass']/max(parse_ok,1):.1f}% of parseable)")
    L.append(f"- rows with >=1 execution-derived (D) fact: "
             f"{stats.get('exec_rows_with_facts', 0)} "
             f"({100*stats.get('exec_rows_with_facts',0)/max(parse_ok,1):.1f}% of parseable)")
    L.append(f"- leak-dropped facts: {stats['leak_dropped']} | "
             f"stacks built: {n_stacks} (size {STACK_SIZE}) | "
             f"C5 distractor pool: {len(pools['doc_first_lines'])} docstring lines")
    L.append("")
    L.append("## Per-intent supply")
    L.append("")
    L.append("| intent | group | applicable fns | triples | proj. full corpus | flags | description |")
    L.append("|---|---|---|---|---|---|---|")
    for iid in intents_sorted:
        spec = REGISTRY[iid]
        proj = em.get(iid, 0) / max(n_rows, 1) * 2_314_889
        flags = []
        if proj < 5000:
            flags.append("LOW-SUPPLY")
        if spec.held_out:
            flags.append("HELD-OUT")
        if spec.needs_luna:
            flags.append("needs-luna")
        if iid in ("C2",):
            flags.append("luna/exec split")
        if iid in SPEC_NOTES:
            flags.append("spec_question")
        if spec.kind == "stack":
            flags.append("stack; needs-phrasing")
        if spec.kind == "exec":
            flags.append("exec")
        L.append(f"| {iid} | {spec.group} | {ap.get(iid, 0)} | {em.get(iid, 0)} "
                 f"| {proj:,.0f} | {', '.join(flags)} | {spec.description} |")
    L.append("")
    L.append("## Per-group supply vs quota")
    L.append("")
    L.append("| group | quota % of mix | triples | share of emitted % | verdict |")
    L.append("|---|---|---|---|---|")
    for g, q in GROUP_QUOTAS.items():
        t = by_group[g][1]
        share = 100 * t / total_em
        # supply is quota-feasible if the group's per-row supply, projected to
        # the full corpus, exceeds its quota share of a 400k-pair target mix
        proj = t / max(n_rows, 1) * 2_314_889
        need = q / 100 * 400_000
        verdict = "OK" if proj >= need else f"SHORT (proj {proj:,.0f} < need {need:,.0f})"
        L.append(f"| {g} | {q}% | {t} | {share:.1f}% | {verdict} |")
    L.append("")
    L.append("Quota check: projected full-corpus supply (per-row rate x 2.31M rows) "
             "vs the group's share of a 400k-pair training mix. Rebalancing across "
             "intents within a group is the sampler's job, not the extractor's.")
    L.append("")
    L.append("## Special supplies")
    hold = sum(em.get(i, 0) for i in ("F2", "D7"))
    luna = sum(1 for t in triples_all if t["needs_luna"])
    phra = sum(1 for t in triples_all if t["needs_phrasing"])
    absn = em.get("B6", 0) + em.get("I2", 0)
    i5_no = sum(1 for t in triples_all if t["intent"] == "I5" and t["answer"] == "no")
    L.append(f"- hold-outs: F2={em.get('F2',0)}, D7={em.get('D7',0)} "
             f"(total {hold}; never trained, eval-only)")
    L.append(f"- needs_luna slots (C1, C3, C2-not-runnable): {luna}")
    L.append(f"- needs_phrasing seeds (group I): {phra}")
    L.append(f"- absence negatives: B6={em.get('B6',0)}, I2={em.get('I2',0)} "
             f"(total {absn}); I5 'no' cases: {i5_no}")
    L.append("")
    L.append("## Execution engine")
    L.append("- isolation: **subprocess** (`python -I` child, restricted "
             "builtins and module allowlist, RLIMIT_CPU 10s, RLIMIT_AS 512MB "
             "where supported, per-call 2s timer, 200k trace-event cap, and "
             "parent-side timeout with row-level retry). The AST screen is the "
             "primary safety layer (see qa_exec.py).")
    L.append(f"- screen pass: {stats['screen_pass']}/{parse_ok} parseable "
             f"({100*stats['screen_pass']/max(parse_ok,1):.1f}%)")
    top_reasons = stats["screen_reasons"].most_common(8)
    L.append("- top screen-reject reasons: "
             + ", ".join(f"{r} {c}" for r, c in top_reasons))
    L.append("- sandbox batch status: "
             + ", ".join(f"{k}={v}" for k, v in sorted(stats["exec_status"].items())))
    L.append("- per-input run status: "
             + ", ".join(f"{k}={v}" for k, v in sorted(stats["run_status"].items())))
    L.append("")
    L.append("## Determinism & leak prevention")
    L.append("- all rng seeded by md5(qualified); double-execution drops nondet "
             "runs (see run status above); sample selection seeded.")
    L.append("- leak rule: non-MCQ seeds must not contain the answer "
             f"(len>=3, non-trivial tokens); {stats['leak_dropped']} facts dropped. "
             "By-design seed/answer pairings (not leaks): D2 carries the OUTPUT "
             "in the seed (answer = input), C2-exec carries the input args. "
             "D1/D3 additionally drop facts whose answer value appears verbatim "
             "in the seed's input string (None/True/echoes), which the trivial-"
             "token exemption would otherwise let through. "
             "No intent was found where leakage is structurally unavoidable. "
             "Residual known shortcut: small-integer answers (D6 counts, A2/A11) "
             "can coincide with digits in seeds/inputs; the phrasing-stage "
             "leak filter is the second gate for those.")
    L.append("")
    L.append("## Distractor recipes (MCQ intents)")
    for iid, rec in DISTRACTOR_RECIPES.items():
        L.append(f"- {iid}: {rec}")
    L.append("")
    L.append("## Spec questions (catalog ambiguities, implemented best reading)")
    for iid, note in SPEC_NOTES.items():
        L.append(f"- {iid}: {note}")
    L.append("- A4/A6/A7/B2/B3/E1 cap answer length (80-200 chars) to keep "
             "answers gradeable; capped-out cases counted as applicable, not emitted.")
    L.append("- F3 treats underscore-prefixed params/locals as intentionally "
             "unused (do not trigger 'yes').")
    L.append("- D3 pins 'value after line N' to the FIRST execution of N "
             "(loops make 'after' ambiguous); seed carries occurrence=first.")
    L.append("")
    L.append("## Timing")
    for k, v in timings.items():
        L.append(f"- {k}: {v:.1f}s")
    total_t = sum(timings.values())
    rate = n_rows / max(total_t, 1e-9)
    L.append(f"- total: {total_t:.1f}s ({rate:,.0f} rows/s) -> projected full "
             f"corpus (2.31M rows): {2_314_889/max(rate,1e-9)/60:.0f} min at "
             f"{args.workers} workers")
    L.append("")
    L.append("## Examples")
    L.append("20 random triples per group are written to "
             "data/corpus_v2/qa_meta/examples/ with their code.")
    L.append("")
    Path(path).write_text("\n".join(L))


def dump_examples(triples, rows_by_idx, out_dir):
    ex_dir = Path(out_dir) / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SAMPLE_SEED)
    by_group = defaultdict(list)
    for t in triples:
        by_group[t["group"]].append(t)
    for g, ts in sorted(by_group.items()):
        picks = rng.sample(ts, min(20, len(ts)))
        fname = ex_dir / f"group_{g.replace('/', '')}.jsonl"
        with open(fname, "w") as f:
            for t in picks:
                rec = dict(t)
                row = rows_by_idx.get(t["row_idx"])
                rec["code"] = (row["code"][:2000] if row else None)
                f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_v2/corpus")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out-dir", default="data/corpus_v2/qa_meta")
    ap.add_argument("--triples-name", default="triples_dev.jsonl")
    ap.add_argument("--report", default="out/qa/COVERAGE_REPORT.md")
    ap.add_argument("--no-exec", action="store_true")
    args = ap.parse_args()

    from datasets import load_from_disk

    t0 = time.perf_counter()
    ds = load_from_disk(args.corpus)[args.split]
    n = len(ds)
    if args.limit and args.limit < n:
        idxs = sorted(random.Random(SAMPLE_SEED).sample(range(n), args.limit))
    else:
        idxs = list(range(n))
    keep = ["repo", "path", "func_name", "code", "n_tokens", "qualified",
            "is_method", "is_async", "decorated", "has_docstring", "typed",
            "string_share", "n_strings"]
    # direct pyarrow access: ds.select + per-column access re-resolves the
    # indices mapping per column (~50 rows/s); take() overflows 32-bit string
    # offsets on the 7GB code column; per-element chunked-array access is fast
    tbl = ds.data.table if hasattr(ds.data, "table") else ds.data
    cols = {k: tbl.column(k) for k in keep}
    rows = [{k: cols[k][i].as_py() for k in keep} for i in idxs]
    rows_by_idx = dict(zip(idxs, rows))
    timings = {"load": time.perf_counter() - t0}
    print(f"[load] {len(rows)} rows in {timings['load']:.1f}s", flush=True)

    t0 = time.perf_counter()
    pools = build_pools(rows, args.workers)
    timings["pools"] = time.perf_counter() - t0
    print(f"[pools] {len(pools['doc_first_lines'])} docstring lines "
          f"in {timings['pools']:.1f}s", flush=True)

    t0 = time.perf_counter()
    stats = {
        "rows": 0, "parse_fail": 0, "normalized": 0, "leak_dropped": 0,
        "applicable": Counter(), "emitted": Counter(), "screen_pass": 0,
        "screen_reasons": Counter(), "exec_status": Counter(),
        "run_status": Counter(),
    }
    pairs = list(zip(idxs, rows))
    chunks = [pairs[i:i + 250] for i in range(0, len(pairs), 250)]
    triples, exec_tasks, ok_idxs = [], [], []
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(pools,)) as p:
        for tr, st, et, ok in p.imap_unordered(_extract_worker, chunks):
            triples.extend(tr)
            exec_tasks.extend(et)
            ok_idxs.extend(ok)
            for k in ("rows", "parse_fail", "normalized", "leak_dropped",
                      "screen_pass"):
                stats[k] += st[k]
            stats["applicable"].update(st["applicable"])
            stats["emitted"].update(st["emitted"])
            stats["screen_reasons"].update(st["screen_reasons"])
    timings["extract"] = time.perf_counter() - t0
    print(f"[extract] {len(triples)} triples, {stats['screen_pass']} screen-pass "
          f"in {timings['extract']:.1f}s", flush=True)

    t0 = time.perf_counter()
    stacks = build_stacks(rows_by_idx, ok_idxs)
    triples.extend(extract_stacks(stacks, rows_by_idx, stats))
    timings["stacks"] = time.perf_counter() - t0
    print(f"[stacks] {len(stacks)} stacks in {timings['stacks']:.1f}s", flush=True)

    t0 = time.perf_counter()
    if not args.no_exec:
        exec_tasks.sort(key=lambda t: t["qualified"])
        triples.extend(run_exec_stage(exec_tasks, args.workers, stats))
    timings["exec"] = time.perf_counter() - t0
    print(f"[exec] {len(exec_tasks)} tasks in {timings['exec']:.1f}s", flush=True)

    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / args.triples_name, "w") as f:
        for t in triples:
            f.write(json.dumps(t) + "\n")
    dump_examples(triples, rows_by_idx, out_dir)
    timings["write"] = time.perf_counter() - t0

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_report(args.report, args, stats, triples, len(rows), timings,
                 len(stacks), pools)
    print(f"[done] {len(triples)} triples -> {out_dir / args.triples_name}; "
          f"report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
