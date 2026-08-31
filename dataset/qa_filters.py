"""QA phrasing pipeline — post-generation filter pass.

Consumes qa_raw.jsonl (dataset/qa_phrase.py `phrase` output) and produces the
final trainer-facing qa_pairs_*.jsonl shards plus FILTER_REPORT.md.

Stages, in order (each drop is counted per intent, nothing dies silently):
  1. format sanity   — repair trivial generation artifacts, then bound length,
                       single line, non-empty, no meta-references.
  2. leak filter     — the answer (case-folded / whitespace-collapsed /
                       quote-unified variants) must not appear in the
                       question. yesno + letter answers are exempt (binary /
                       A-D tokens appear naturally); number answers are
                       checked as standalone tokens. extra_leak_terms
                       (I-group function names) are checked too.
  3. phrasing dedup  — per intent, ROUGE-L > 0.85 against the accepted
                       pool assigns a record to a phrasing cluster; each
                       cluster keeps at most --dup-cap records (default 25).
                       --dup-cap 1 = strict catalog reading (one record per
                       phrasing). Values above one preserve supply across
                       different functions while limiting surface collapse.
  4. round-trip      — the same vLLM endpoint answers the question WITH code
                       visible; extractive containment / normalized match
                       against ground truth. Failures are FATAL only for
                       needs_roundtrip=="high" records (I-group/stack: the
                       round trip tests the QUESTION). Elsewhere, the answer
                       is construction-verified upstream; the verdict is
                       recorded as rt_pass in metadata and failures remain
                       advisory. --mock-judge exercises the code path with a
                       deterministic synthetic verdict. --skip-roundtrip
                       omits the stage and records that choice.
  5. assembly        — held-out records (F2/D7) pass through with
                       held_out=true; contract-only fields go to
                       qa_pairs_XXXX.jsonl; provenance (style, template,
                       qualified, needs_roundtrip) to
                       qa_pairs_meta_XXXX.jsonl; rejected records' pair_id +
                       drop stage to qa_rejects_meta.jsonl;
                       every count to FILTER_REPORT.md. All shards are
                       validated with validate_record() before the report is
                       written.

Usage:
  uv run python dataset/qa_filters.py --out-dir data/corpus_v2/qa_phrased \
      --mock-judge            # local mock test
  uv run python dataset/qa_filters.py --out-dir ... --roundtrip \
      --concurrency 256
  uv run python dataset/qa_filters.py --validate-only path/to/qa_pairs_0000.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qa_phrase import (HELD_OUT_INTENTS,  # noqa: E402,F401  (shared)
                       TRIVIAL_LEAK_EXEMPT)

SHARD_SIZE = 100_000

CONTRACT_ANSWER_TYPES = {"substring", "number", "yesno", "name", "letter",
                         "line_ref"}
CONTRACT_SOURCES = {"ast", "exec", "luna", "mcq"}


# --------------------------------------------------------------------------
# Contract checker (the trainer consumer builds to exactly this schema)
# --------------------------------------------------------------------------

def validate_record(rec: dict) -> list[str]:
    """Return a list of contract violations (empty = valid)."""
    errs = []

    def need(k, types):
        if k not in rec:
            errs.append(f"missing {k}")
            return None
        if not isinstance(rec[k], types):
            errs.append(f"{k} wrong type {type(rec[k]).__name__}")
            return None
        return rec[k]

    need("row_idx", int)
    it = need("intent", str)
    ho = need("held_out", bool)
    if ho is not None and it is not None and ho != (it in HELD_OUT_INTENTS):
        # never trust the upstream flag alone: a stale/malformed triple could
        # leak a held-out intent into training pools
        errs.append("held_out flag mismatches intent")
    q = need("question", str)
    if q is not None and not q.strip():
        errs.append("question empty")
    a = need("answer", str)
    if a is not None and not a.strip():
        # whitespace-only answers normalize to "" and then ANY prediction
        # passes containment grading
        errs.append("answer blank")
    at = need("answer_type", str)
    if at is not None and at not in CONTRACT_ANSWER_TYPES:
        errs.append(f"answer_type {at!r} not in contract")
    src = need("source", str)
    if src is not None and src not in CONTRACT_SOURCES:
        errs.append(f"source {src!r} not in contract")
    ctx = need("context", dict)
    if ctx is not None:
        kind = ctx.get("kind")
        if kind not in ("solo", "stack"):
            errs.append(f"context.kind {kind!r}")
        if kind == "stack":
            if not (isinstance(ctx.get("stack_row_idxs"), list)
                    and ctx["stack_row_idxs"]
                    and all(isinstance(i, int)
                            for i in ctx["stack_row_idxs"])):
                errs.append("context.stack_row_idxs bad")
            if not isinstance(ctx.get("target_row_idx"), int):
                errs.append("context.target_row_idx bad")
        if kind == "solo":
            for k in ("stack_row_idxs", "target_row_idx"):
                if k in ctx:
                    errs.append(f"context.{k} present on solo")
    if src == "mcq":
        # rt rendering indexes a fixed "ABCD" — 5+ options would crash it,
        # and substring letter checks let "AB" through
        opts = rec.get("options")
        if not (isinstance(opts, list)
                and 2 <= len(opts) <= 4
                and all(isinstance(o, str) and o.strip() for o in opts)
                and len({_norm(o) for o in opts}) == len(opts)):
            errs.append("options bad/missing on mcq")
        elif (at == "letter" and a in ("A", "B", "C", "D")
                and ord(a) - 65 >= len(opts)):
            # answer letter must index an existing option: "D" with two
            # options passed the old check
            errs.append(f"letter answer {a!r} beyond {len(opts)} options")
    elif "options" in rec:
        errs.append("options present on non-mcq")
    if at == "letter" and a is not None and a not in ("A", "B", "C", "D"):
        errs.append(f"letter answer {a!r}")
    if (it == "I6" and at == "name" and isinstance(a, str) and "," in a):
        # I6 uses set grading, so expected names must remain distinct under
        # normalization. Other comma-separated answers may contain ordered
        # expressions and do not use this distinctness requirement.
        parts = [p.strip() for p in a.split(",") if p.strip()]
        if len({_norm(p) for p in parts}) != len(parts):
            errs.append("multi-name answer collapses under normalization")
    if at == "yesno" and a is not None and a not in ("yes", "no"):
        errs.append(f"yesno answer {a!r}")
    return errs


def to_contract(rec: dict) -> dict:
    out = {
        "row_idx": rec["row_idx"],
        "intent": rec["intent"],
        "held_out": rec["held_out"],
        "question": rec["question"],
        "answer": rec["answer"],
        "answer_type": rec["answer_type"],
        "context": rec["context"],
        "source": rec["source"],
    }
    if rec["source"] == "mcq":
        out["options"] = rec["options"]
    return out


# --------------------------------------------------------------------------
# 1. format sanity
# --------------------------------------------------------------------------

_META_RX = re.compile(
    r"fact card|ground.truth|style directive|as an ai|i cannot|the answer is",
    re.IGNORECASE)


def repair(q: str) -> str:
    q = re.sub(r"<think>.*?</think>", "", q, flags=re.S)
    q = q.strip().strip("`").strip()
    for pfx in ("Question:", "question:", "Q:"):
        if q.startswith(pfx):
            q = q[len(pfx):].strip()
    if len(q) >= 2 and q[0] in "\"'" and q[-1] == q[0]:
        q = q[1:-1].strip()
    return " ".join(q.split())


def format_reject(q: str) -> str | None:
    if not q:
        return "empty"
    if len(q) < 8:
        return "too_short"
    if len(q) > 500:
        return "too_long"
    if not re.search(r"[A-Za-z]", q):
        return "no_letters"
    if "```" in q:
        return "code_fence"
    if _META_RX.search(q):
        return "meta_reference"
    return None


# --------------------------------------------------------------------------
# 2. leak filter
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("'", '"')
    return " ".join(s.lower().split())


def leaks(question: str, answer: str, answer_type: str,
          extra_terms: list[str]) -> bool:
    qn = _norm(question)
    terms = list(extra_terms or [])
    if (answer_type not in ("yesno", "letter")
            and _norm(str(answer)) not in TRIVIAL_LEAK_EXEMPT):
        terms.append(answer)
        # multi-name answers ("name_a, name_b" — I6/list types): each component
        # must be absent individually, else name_b alone slips the check
        if answer_type == "name" and "," in str(answer):
            terms.extend(p.strip() for p in str(answer).split(",") if p.strip())
    for t in terms:
        tn = _norm(str(t)).strip('"').strip()
        if not tn:
            continue
        if re.fullmatch(r"-?\d+(\.\d+)?", tn):
            if re.search(rf"(?<![\w.]){re.escape(tn)}(?![\w.])", qn):
                return True
        elif re.fullmatch(r"\w+", tn):
            # single identifier/word (any length): token-boundary match, so
            # a function named 'get' doesn't nuke every question with 'gets'
            if re.search(rf"(?<!\w){re.escape(tn)}(?!\w)", qn):
                return True
        elif len(tn) >= 3:
            if tn in qn:
                return True
        elif not re.search(r"\w", tn):
            # short pure-symbol answers ("[]", "#", "->") previously bypassed
            # the gate entirely: word-boundary match keeps
            # "obj.attr" from tripping on a "." answer while catching
            # "Does it return []?" with answer "[]"
            if re.search(rf"(?<!\w){re.escape(tn)}(?!\w)", qn):
                return True
    return False


# --------------------------------------------------------------------------
# 3. phrasing dedup (ROUGE-L clusters, capped)
# --------------------------------------------------------------------------

def _tokens(q: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", q.lower())


def rouge_l_f(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y
                       else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(b), lcs / len(a)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


class DedupPool:
    """Per-intent phrasing clusters. Exact-hash fast path, then ROUGE-L
    against candidate cluster reps sharing >= 50% tokens (inverted index)."""

    def __init__(self, threshold: float, cap: int):
        self.threshold = threshold
        self.cap = cap
        self.reps: list[list[str]] = []       # cluster representatives
        self.counts: list[int] = []
        self.exact: dict[str, int] = {}       # norm question -> cluster id
        self.inv: dict[str, set[int]] = defaultdict(set)

    def admit(self, q: str) -> bool:
        qn = _norm(q)
        toks = _tokens(q)
        cid = self.exact.get(qn)
        if cid is None:
            cand = Counter()
            for t in set(toks):
                for c in self.inv.get(t, ()):
                    cand[c] += 1
            for c, shared in cand.most_common(50):
                if shared / max(1, min(len(set(toks)),
                                       len(set(self.reps[c])))) < 0.5:
                    break
                if rouge_l_f(toks, self.reps[c]) > self.threshold:
                    cid = c
                    break
        if cid is None:
            cid = len(self.reps)
            self.reps.append(toks)
            self.counts.append(0)
            for t in set(toks):
                self.inv[t].add(cid)
        self.exact[qn] = cid
        if self.counts[cid] >= self.cap:
            return False
        self.counts[cid] += 1
        return True


# --------------------------------------------------------------------------
# 4. round-trip answerability
# --------------------------------------------------------------------------

RT_SYSTEM = ("You answer questions about the Python code you are shown. "
             "Reply with the exact value only — no explanation, no "
             "sentence, just the answer.")


def rt_messages(rec: dict) -> list[dict]:
    code = rec.get("stack_code") or rec["code"]
    parts = [f"Code:\n```python\n{code}\n```"]
    if rec.get("options"):
        letters = "ABCD"
        parts.append("Options:\n" + "\n".join(
            f"{letters[i]}. {o}" for i, o in enumerate(rec["options"])))
        parts.append(f"Question: {rec['question']}\n"
                     f"Answer with the single letter of the correct option.")
    else:
        parts.append(f"Question: {rec['question']}")
    return [{"role": "system", "content": RT_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)}]


def rt_grade(pred: str, rec: dict) -> bool:
    gt = str(rec["answer"])
    at = rec["answer_type"]
    pred = re.sub(r"<think>.*?</think>", "", pred, flags=re.S).strip()
    if at == "yesno":
        m = re.search(r"\b(yes|no)\b", pred.lower())
        return bool(m and m.group(1) == gt)
    if at == "letter":
        m = re.search(r"\b([A-D])\b", pred)
        return bool(m and m.group(1) == gt)
    if at == "number":
        m = re.search(r"-?\d+(?:\.\d+)?", pred)
        if not m:
            return False
        try:
            return float(m.group(0)) == float(gt)
        except ValueError:
            return False
    if at == "line_ref":
        return re.findall(r"\d+", pred) == re.findall(r"\d+", gt)
    # name / substring: normalized equality or containment
    gn, pn = _norm(gt), _norm(pred)
    if not gn:
        # blank ground truth normalizes to "" and "" is contained in every
        # prediction — ungradeable, fail closed
        return False
    if at == "name" and "," in gt:
        parts = [p for p in gt.split(",") if p.strip()]
        want = {_norm(p) for p in parts}
        if rec.get("intent") == "I6":
            if len(want) != len(parts):
                return False  # names collapse under normalization (Add/add)
            # Multi-name I6 answers are sets. Identifier-boundary matching
            # permits reordered answers while rejecting extra sibling names.
            cands = want | {_norm(s) for s in rec.get("sibling_names") or []}
            hit = {n for n in cands
                   if n and re.search(rf"(?<!\w){re.escape(n)}(?!\w)", pn)}
            return bool(pn) and hit == want
        # Other comma-separated name answers, such as A3 parameter lists, are
        # ordered and must not use set semantics.
        return [_norm(p) for p in parts] == \
            [_norm(p) for p in pn.split(",") if p.strip()]
    if at == "name":
        # single names: identifier-boundary containment, not raw substring
        # ("add" was credited inside "add_numbers", and a stacked "foo"
        # answer accepted "foo and bar" with sibling bar)
        if not (pn and re.search(rf"(?<!\w){re.escape(gn)}(?!\w)", pn)):
            return False
        sibs = {_norm(s) for s in rec.get("sibling_names") or []} - {gn}
        return not any(s and re.search(rf"(?<!\w){re.escape(s)}(?!\w)", pn)
                       for s in sibs)
    return bool(pn) and (gn == pn or gn in pn)


def mock_judge_pass(pair_id: str) -> bool:
    return int(hashlib.md5(f"judge|{pair_id}".encode()).hexdigest(),
               16) % 100 < 95


async def roundtrip_llm(records: list[dict], args) -> dict[str, bool]:
    import aiohttp
    url = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B")
    sem = asyncio.Semaphore(args.concurrency)
    verdicts: dict[str, bool] = {}

    async def one(session, rec):
        payload = {"model": model, "messages": rt_messages(rec),
                   "temperature": 0.0, "max_tokens": 200,
                   "chat_template_kwargs": {"enable_thinking": False}}
        async with sem:
            for attempt in range(6):
                try:
                    async with session.post(
                            f"{url}/chat/completions", json=payload,
                            timeout=aiohttp.ClientTimeout(total=180)) as r:
                        if r.status in (429, 500, 502, 503, 504):
                            raise aiohttp.ClientError(f"HTTP {r.status}")
                        r.raise_for_status()
                        data = await r.json()
                    pred = data["choices"][0]["message"]["content"]
                    verdicts[rec["pair_id"]] = rt_grade(pred, rec)
                    return
                except (aiohttp.ClientError, asyncio.TimeoutError,
                        KeyError) as e:
                    if attempt == 5:
                        verdicts[rec["pair_id"]] = False  # fail closed
                        return
                    await asyncio.sleep(min(60, 2 ** attempt)
                                        + random.random())

    async with aiohttp.ClientSession() as session:
        chunk = 20000
        for i in range(0, len(records), chunk):
            await asyncio.gather(*(one(session, r)
                                   for r in records[i:i + chunk]))
            print(f"[roundtrip] {min(i+chunk, len(records))}/{len(records)}",
                  flush=True)
    return verdicts


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="QA post-generation filters")
    ap.add_argument("--out-dir", default="data/corpus_v2/qa_phrased")
    ap.add_argument("--raw", default=None,
                    help="input (default <out-dir>/qa_raw.jsonl)")
    ap.add_argument("--dup-threshold", type=float, default=0.85)
    ap.add_argument("--dup-cap", type=int, default=25,
                    help="max records per phrasing cluster per intent; "
                         "1 = strict one-per-phrasing")
    ap.add_argument("--roundtrip", action="store_true",
                    help="run round-trip via VLLM_URL")
    ap.add_argument("--mock-judge", action="store_true",
                    help="deterministic ~95%%-pass round-trip stand-in")
    ap.add_argument("--skip-roundtrip", action="store_true",
                    help="explicitly select skipped mode (same behavior as "
                         "the default; exists so runbooks state intent)")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--validate-only", default=None,
                    help="validate an existing qa_pairs shard and exit")
    args = ap.parse_args()
    if sum(map(bool, (args.roundtrip, args.mock_judge,
                      args.skip_roundtrip))) > 1:
        # --skip-roundtrip was parsed but never consulted, so combining it
        # with --mock-judge silently ran the mock
        ap.error("--roundtrip / --mock-judge / --skip-roundtrip are "
                 "mutually exclusive")

    if args.validate_only:
        bad = 0
        with open(args.validate_only) as f:
            for ln, line in enumerate(f, 1):
                errs = validate_record(json.loads(line))
                if errs:
                    bad += 1
                    print(f"line {ln}: {errs}")
        print(f"{'FAIL' if bad else 'PASS'}: {bad} invalid records")
        sys.exit(1 if bad else 0)

    out = Path(args.out_dir)
    raw_path = Path(args.raw) if args.raw else out / "qa_raw.jsonl"
    if not raw_path.exists():
        sys.exit(f"{raw_path} missing")

    records = []
    with open(raw_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[filters] {len(records):,} raw records from {raw_path}")
    # Async completion order correlates with prompt length and would bias
    # capped deduplication clusters. Use deterministic content-keyed order.
    records.sort(
        key=lambda r: hashlib.md5(str(r.get("pair_id", "")).encode()).hexdigest())

    drops = defaultdict(Counter)      # stage -> intent counter
    drop_reasons = Counter()
    planted_total = sum(1 for r in records if r.get("planted_leak"))
    planted_killed = 0

    # 1+2: format sanity + leak filter -----------------------------------
    survivors = []
    for r in records:
        q = repair(r["question"])
        r["question"] = q
        why = format_reject(q)
        if why:
            drops["format"][r["intent"]] += 1
            drop_reasons[f"format:{why}"] += 1
            r["_drop"] = f"format:{why}"
            continue
        if leaks(q, r["answer"], r["answer_type"],
                 r.get("extra_leak_terms")):
            drops["leak"][r["intent"]] += 1
            if r.get("planted_leak"):
                planted_killed += 1
            r["_drop"] = "leak"
            continue
        survivors.append(r)
    print(f"[filters] format+leak: {len(survivors):,} pass "
          f"({sum(drops['format'].values())} format, "
          f"{sum(drops['leak'].values())} leak; planted killed "
          f"{planted_killed}/{planted_total})")

    # 3: per-intent phrasing dedup ---------------------------------------
    t0 = time.perf_counter()
    pools: dict[str, DedupPool] = {}
    deduped = []
    for r in survivors:
        pool = pools.setdefault(
            r["intent"], DedupPool(args.dup_threshold, args.dup_cap))
        if pool.admit(r["question"]):
            deduped.append(r)
        else:
            drops["dedup"][r["intent"]] += 1
            r["_drop"] = "dedup"
    n_clusters = {i: len(p.reps) for i, p in pools.items()}
    print(f"[filters] dedup: {len(deduped):,} pass "
          f"({sum(drops['dedup'].values())} dropped, "
          f"{sum(n_clusters.values())} phrasing clusters) "
          f"in {time.perf_counter()-t0:.1f}s")

    # 4: round-trip ------------------------------------------------------
    rt_mode = ("llm" if args.roundtrip else
               "mock" if args.mock_judge else "skipped")
    if rt_mode == "llm":
        verdicts = asyncio.run(roundtrip_llm(deduped, args))
    elif rt_mode == "mock":
        verdicts = {r["pair_id"]: mock_judge_pass(r["pair_id"])
                    for r in deduped}
    else:
        verdicts = {}
        print("[filters] WARNING: round-trip SKIPPED — do not train on "
              "this output without a round-trip pass")
    # Drop policy: an rt failure is fatal only where the round-trip tests the
    # QUESTION (uniqueness/answerability): I-group and stack records, marked
    # needs_roundtrip == "high" at build time. Everywhere else the answer is
    # construction-verified upstream (AST / sandbox execution / gated luna) and
    # a failure indicts the string-match judge, not the data — the verdict is
    # recorded (rt_pass in meta) but the record is kept. Skipped mode records
    # rt_pass=null (no judgment != a pass).
    final = []
    advisory = Counter()
    held_advisory = Counter()
    for r in deduped:
        if rt_mode == "skipped":
            r["_rt_pass"] = None
            final.append(r)
            continue
        ok = verdicts.get(r["pair_id"], False)
        r["_rt_pass"] = bool(ok)
        if ok:
            final.append(r)
        elif r.get("needs_roundtrip") != "high":
            advisory[r["intent"]] += 1
            if r.get("held_out"):
                held_advisory[r["intent"]] += 1
            final.append(r)
        else:
            drops["roundtrip"][r["intent"]] += 1
            r["_drop"] = "roundtrip"
    n_advisory = sum(advisory.values())
    print(f"[filters] round-trip ({rt_mode}): {len(final):,} pass "
          f"({sum(drops['roundtrip'].values())} dropped fatal; "
          f"{n_advisory} kept with rt_pass=false)")

    # 5: assembly + validation -------------------------------------------
    # A smaller rerun must never leave higher-numbered shards from a previous
    # pass for the trainer glob to silently mix in.
    stale = sorted(out.glob("qa_pairs_*.jsonl"))
    for sp in stale:
        sp.unlink()
    if stale:
        print(f"[filters] removed {len(stale)} pre-existing qa_pairs*.jsonl "
              f"shards (pair + meta) before writing")
    n_invalid = 0
    shard, in_shard = -1, SHARD_SIZE
    pf = mf = None
    held_counts = Counter()
    intent_counts = Counter()
    for r in final:
        rec = to_contract(r)
        errs = validate_record(rec)
        if errs:
            n_invalid += 1
            drops["contract"][r["intent"]] += 1
            drop_reasons[f"contract:{errs[0]}"] += 1
            r["_drop"] = f"contract:{errs[0]}"
            continue
        if in_shard >= SHARD_SIZE:
            if pf:
                pf.close(); mf.close()
            shard += 1
            in_shard = 0
            pf = open(out / f"qa_pairs_{shard:04d}.jsonl", "w")
            mf = open(out / f"qa_pairs_meta_{shard:04d}.jsonl", "w")
        pf.write(json.dumps(rec) + "\n")
        mf.write(json.dumps({
            "pair_id": r["pair_id"], "qualified": r["qualified"],
            "style_id": r.get("style_id"),
            "template_id": r.get("template_id"),
            "needs_roundtrip": r.get("needs_roundtrip"),
            "rt_pass": r.get("_rt_pass"),
            "mock": r.get("mock", False),
            "gen_model": r.get("gen_model"),
        }) + "\n")
        in_shard += 1
        intent_counts[r["intent"]] += 1
        if rec["held_out"]:
            held_counts[r["intent"]] += 1
    if pf:
        pf.close(); mf.close()
    n_shards = shard + 1
    # Write auditable per-record reject provenance outside every qa_pairs_* glob
    # so consumers see only pair and metadata shards.
    n_rej = 0
    with open(out / "qa_rejects_meta.jsonl", "w") as rf:
        for r in records:
            if "_drop" in r:
                rf.write(json.dumps({"pair_id": r.get("pair_id"),
                                     "intent": r.get("intent"),
                                     "drop": r["_drop"]}) + "\n")
                n_rej += 1
    print(f"[filters] wrote {sum(intent_counts.values()):,} pairs in "
          f"{n_shards} shards ({n_invalid} contract-invalid dropped); "
          f"held-out passthrough: {dict(held_counts)}; "
          f"{n_rej:,} rejects -> qa_rejects_meta.jsonl")

    write_report(out, args, len(records), drops, drop_reasons,
                 intent_counts, held_counts, n_clusters, rt_mode,
                 planted_total, planted_killed, n_shards, advisory,
                 held_advisory)


def write_report(out, args, n_raw, drops, drop_reasons, intent_counts,
                 held_counts, n_clusters, rt_mode, planted_total,
                 planted_killed, n_shards, advisory, held_advisory):
    L = ["# QA phrasing — filter report\n",
         f"Input: {n_raw:,} raw records. Output: "
         f"{sum(intent_counts.values()):,} pairs in {n_shards} shards. "
         f"Round-trip mode: **{rt_mode}**. dup-cap {args.dup_cap}, "
         f"ROUGE-L threshold {args.dup_threshold}.\n"]
    if sum(advisory.values()):
        L.append(f"**Round-trip advisory: {sum(advisory.values()):,} records "
                 f"RETAINED with rt_pass=false** (policy: rt failure fatal "
                 f"only for needs_roundtrip=high; every retained failure is "
                 f"flagged in meta). Held-out among them: "
                 f"{sum(held_advisory.values()):,} "
                 f"({dict(held_advisory) or '{}'}) — exclude from, or "
                 f"semantically regrade for, the held-out QA eval.\n")
    if planted_total:
        rate = planted_killed / planted_total
        L.append(f"**Planted-leak check: {planted_killed}/{planted_total} "
                 f"killed ({rate:.0%}).**\n")
    L.append("## Drops by stage\n")
    L.append("| stage | dropped |")
    L.append("|---|---|")
    for stage in ("format", "leak", "dedup", "roundtrip", "contract"):
        L.append(f"| {stage} | {sum(drops[stage].values()):,} |")
    L.append("\n## Per-intent\n")
    L.append("| intent | kept | format | leak | dedup | roundtrip | "
             "rt_false | contract | clusters | held_out |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    all_intents = sorted(set(intent_counts) | set(n_clusters) | set(advisory)
                         | {i for s in drops.values() for i in s})
    for i in all_intents:
        L.append(f"| {i} | {intent_counts.get(i, 0):,} "
                 f"| {drops['format'].get(i, 0)} "
                 f"| {drops['leak'].get(i, 0)} "
                 f"| {drops['dedup'].get(i, 0)} "
                 f"| {drops['roundtrip'].get(i, 0)} "
                 f"| {advisory.get(i, 0)} "
                 f"| {drops['contract'].get(i, 0)} "
                 f"| {n_clusters.get(i, 0)} "
                 f"| {held_counts.get(i, 0)} |")
    if drop_reasons:
        L.append("\n## Fine-grained drop reasons\n")
        for k, v in drop_reasons.most_common():
            L.append(f"- {k}: {v}")
    L.append("\n## Configuration notes\n")
    L.append("- dedup is cap-based. A value of one keeps a single record per "
             "phrasing cluster; larger values preserve supply while limiting "
             "surface collapse.")
    L.append("- answer_type maps list to name/substring and text to substring "
             "(see qa_phrase.map_answer_type).")
    if rt_mode != "llm":
        L.append("- round-trip did not run against a model; rerun with "
                 "--roundtrip against the configured endpoint before training.")
    (out / "FILTER_REPORT.md").write_text("\n".join(L) + "\n")
    print(f"[filters] report -> {out/'FILTER_REPORT.md'}")


if __name__ == "__main__":
    main()
