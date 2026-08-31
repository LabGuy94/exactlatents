"""I6 multi-match retrieval supply — skeleton-family pair stacks.

Catalog I6 (out/qa/INTENT_CATALOG.md group I): "which functions do X?"
where 2+ stack members GENUINELY match; answer = both names. Ground truth by
construction: the two matches come from the SAME skeleton family (identical
tokenize-normalized skeleton, annotate.skeleton_hash — near-identical code
modulo identifiers/strings/numbers), from DIFFERENT repos, with DISTINCT short
function names (clean grading). The remaining 6 stack members are fillers drawn
from OTHER families (pairwise-distinct families), so exactly the pair matches.

Skeleton-family identity alone is insufficient: structurally similar functions
can have different behavior in the identifier, string, and number tokens that
the skeleton normalizes away. Candidate pairs therefore require content-token
similarity over NAME, STRING, and NUMBER tokens with keywords, docstrings, and
the function's own name excluded. Each member is paired with its best compatible
partner.

The similarity floor depends on the shorter side: at least 20 content tokens
uses 0.65, 8–19 tokens uses 0.75, and fewer than 8 tokens requires 0.90. This
prevents short delegation boilerplate from qualifying when only the callee
differs. Fillers are also rejected when their content-token Jaccard similarity
to the target reaches ``FILLER_MAX_JACCARD``. The high-priority round-trip check
remains the final disambiguation step.

The generator follows the group-I conventions: eight functions per stack,
md5-qualified ordering, unique short names, deterministic seeds, and
``describe_row_idx`` / ``stack_row_idxs`` phrasing metadata. Its triple schema
matches ``qa_extract.fact_to_triple``.

Row-index mapping follows draw/materialize: annotation parquets are concatenated
in sorted glob order and stable-sorted by ``(pile, shard, line)``. A training
``row_idx`` is the position among ``status == "kept_train"`` rows in that order.

Output is ``data/corpus_v2/qa_full/triples_i6.jsonl`` with 15 inspection
examples under ``data/corpus_v2/qa_meta/examples/group_I6.jsonl``. Pass the
finalized directory to ``qa_phrase`` rather than a broad ``triples_*.jsonl``
glob, which can include raw or usage intermediates.

Run: uv run python dataset/qa_i6_multimatch.py [--target 12000] [--verify 30]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import keyword
import random
import sys
import time
import tokenize
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore", category=SyntaxWarning)

from qa_extract import ParseFailure, make_ctx  # noqa: E402
from annotate import skeleton_hash  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data/corpus_v2/corpus_meta"
ANNOT = META / "annot"
OUT_TRIPLES = ROOT / "data/corpus_v2/qa_full/triples_i6.jsonl"
OUT_EXAMPLES = ROOT / "data/corpus_v2/qa_meta/examples/group_I6.jsonl"

SEED = 20260806
STACK_SIZE = 8
PAIRS_PER_FAMILY_CAP = 16
MIN_PAIR_SIM = 0.65           # content-token floor, >= SHORT_LEN tokens
MIN_PAIR_SIM_SHORT = 0.75     # floor for MIN_CONTENT_TOKENS..SHORT_LEN
MIN_PAIR_SIM_TINY = 0.90      # under MIN_CONTENT_TOKENS: near-identity only
SHORT_LEN = 20                # content tokens; below this the strict floor
MIN_CONTENT_TOKENS = 8        # below this the near-identity floor
FILLER_MAX_JACCARD = 0.5      # filler-vs-target content-token set overlap cap
FILLER_POOL_SIZE = 60_000     # candidate filler rows (parse-screened)
BANDS = [(16, 63), (64, 255), (256, 511), (512, 1023), (1024, 4096)]
BAND_SHARE = [0.10, 0.60, 0.20, 0.08, 0.02]   # draw band mix

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def md5x(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def short(name: str) -> str:
    return name.split(".")[-1]


def content_tokens(code: str, own_name: str) -> list[str]:
    """NAME/STRING/NUMBER tokens in order — keywords out, docstrings/bare
    string statements out, own function name masked to 'F'. This is the
    semantic payload the skeleton hash normalizes away."""
    try:
        out = []
        at_stmt_start = True
        for tk in tokenize.generate_tokens(io.StringIO(code).readline):
            t = tk.type
            if t in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                     tokenize.DEDENT):
                at_stmt_start = True
                continue
            if t == tokenize.STRING and at_stmt_start:
                at_stmt_start = False
                continue
            at_stmt_start = False
            if t == tokenize.NAME and not keyword.iskeyword(tk.string):
                out.append("F" if tk.string == own_name else tk.string)
            elif t in (tokenize.STRING, tokenize.NUMBER):
                out.append(tk.string)
        return out
    except Exception:
        return [w for w in code.split() if w.isidentifier()]


def pair_sim(toks_a: list[str], toks_b: list[str]) -> float:
    return difflib.SequenceMatcher(None, toks_a, toks_b).ratio()


def _floor_for(shorter_side_tokens: int) -> float:
    """Graded genuine-twin floor: the less semantic content a pair carries,
    the closer to identical it must be for 'any description of a matches b'
    to hold."""
    if shorter_side_tokens < MIN_CONTENT_TOKENS:
        return MIN_PAIR_SIM_TINY
    if shorter_side_tokens < SHORT_LEN:
        return MIN_PAIR_SIM_SHORT
    return MIN_PAIR_SIM


# --------------------------------------------------------------------------
# 1. row_idx mapping + family assignments (annot -> corpus train split)
# --------------------------------------------------------------------------

def load_kept_meta():
    """Returns kept-row arrays aligned to train-split row_idx 0..N-1:
    fam codes (int), repo, band, stub — plus the fam-code -> count map."""
    m = pd.read_parquet(META / "draw_manifest.parquet")
    parts = sorted(ANNOT.glob("p*_s*.parquet"))
    log(f"loading {len(parts)} annot parquets ...")
    df = pd.concat(
        [pd.read_parquet(p, columns=["pile", "shard", "line", "fam", "repo", "stub"])
         for p in parts], ignore_index=True)
    df.sort_values(["pile", "shard", "line"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    assert len(df) == len(m), f"annot {len(df)} vs manifest {len(m)}"
    for c in ("pile", "shard", "line"):
        assert (df[c].to_numpy() == m[c].to_numpy()).all(), f"misaligned on {c}"
    kept = (m.status == "kept_train").to_numpy()
    n_kept = int(kept.sum())
    log(f"pool {len(df)} rows, kept_train {n_kept}")

    fam_codes, _ = pd.factorize(df.fam[kept])          # int code per kept row
    repo = df.repo.to_numpy()[kept]
    stub = df.stub.to_numpy()[kept]
    band = m.band.to_numpy()[kept]
    return fam_codes, repo, band, stub


# --------------------------------------------------------------------------
# 2. corpus column access
# --------------------------------------------------------------------------

class Corpus:
    def __init__(self):
        from datasets import load_from_disk
        self.ds = load_from_disk(str(ROOT / "data/corpus_v2/corpus"))["train"]
        self.tbl = self.ds.data.table if hasattr(self.ds.data, "table") else self.ds.data
        self._cols = {}

    def __len__(self):
        return len(self.ds)

    def col(self, name):
        if name not in self._cols:
            self._cols[name] = self.tbl.column(name)
        return self._cols[name]

    def fetch(self, name: str, idxs) -> dict[int, str]:
        c = self.col(name)
        return {int(i): c[int(i)].as_py() for i in idxs}


# --------------------------------------------------------------------------
# 3. family pair selection
# --------------------------------------------------------------------------

def build_pairs(fam_codes, repo, stub, corpus):
    """Qualifying pairs: same family, different repos, distinct short names,
    both non-stub, content-token similarity >= MIN_PAIR_SIM. Within a family
    each member is greedily matched with its BEST-similarity compatible
    partner (disjoint; <= PAIRS_PER_FAMILY_CAP pairs/family). Deterministic:
    md5(qualified) ordering + sim/md5 tie-breaks."""
    order = np.argsort(fam_codes, kind="stable")
    codes_sorted = fam_codes[order]
    starts = np.flatnonzero(np.r_[True, codes_sorted[1:] != codes_sorted[:-1]])
    ends = np.r_[starts[1:], len(codes_sorted)]

    fam_members = {}
    cand_rows = []
    n_multi = 0
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        members = [int(i) for i in order[s:e] if not stub[i]]
        # need at least 2 distinct repos to have any chance
        if len(members) < 2 or len({repo[i] for i in members}) < 2:
            continue
        n_multi += 1
        fam_members[int(codes_sorted[s])] = members
        cand_rows.extend(members)
    log(f"families with >=2 non-stub members across >=2 repos: {n_multi} "
        f"({len(cand_rows)} member rows)")

    log("fetching func_name + qualified + code for member rows ...")
    names = corpus.fetch("func_name", cand_rows)
    quals = corpus.fetch("qualified", cand_rows)
    codes = corpus.fetch("code", cand_rows)
    # parse screen HERE so band supply counts are true (post-hoc drops were
    # eating quota in supply-limited bands)
    parse_ok = set()
    for i in cand_rows:
        try:
            make_ctx({"code": codes[i], "qualified": quals[i]}, i)
            parse_ok.add(i)
        except (ParseFailure, SyntaxError, RecursionError, ValueError,
                MemoryError):
            pass
    log(f"member parse screen: {len(parse_ok)}/{len(cand_rows)} ok")
    fam_members = {fc: [i for i in ms if i in parse_ok]
                   for fc, ms in fam_members.items()}
    toks = {i: content_tokens(codes[i], short(names[i])) for i in parse_ok}

    pairs = []            # (fam_code, a_idx, b_idx, sim); a = md5-first member
    fam_pair_counts = Counter()
    n_below_floor = 0
    sim_kept = []
    for fc, members in fam_members.items():
        ms = sorted(members, key=lambda i: md5x(quals[i]))
        # all compatible combos with their similarity
        combos = []
        for ai, a in enumerate(ms):
            na = short(names[a])
            for b in ms[ai + 1:]:
                # casefold: "Add"/"add" collapse under the graders' lowercase
                # normalization — such pairs are ungradeable
                if repo[a] == repo[b] or na.casefold() == short(names[b]).casefold():
                    continue
                shorter = min(len(toks[a]), len(toks[b]))
                s = pair_sim(toks[a], toks[b])
                if s < _floor_for(shorter):
                    n_below_floor += 1
                    continue
                combos.append((s, a, b))
        # greedy best-first disjoint selection, deterministic tie-break
        combos.sort(key=lambda c: (-c[0], md5x(quals[c[1]] + "|" + quals[c[2]])))
        used = set()
        taken = 0
        for s, a, b in combos:
            if taken >= PAIRS_PER_FAMILY_CAP:
                break
            if a in used or b in used:
                continue
            pairs.append((fc, a, b, s))
            used.update((a, b))
            taken += 1
            sim_kept.append(s)
        if taken:
            fam_pair_counts[taken] += 1
    sim_arr = np.array(sim_kept) if sim_kept else np.array([0.0])
    log(f"qualifying pairs: {len(pairs)} from "
        f"{sum(fam_pair_counts.values())} families "
        f"(pairs-per-family histogram {dict(sorted(fam_pair_counts.items()))}; "
        f"{n_below_floor} combos below graded sim floor; kept sim "
        f"p10/p50/p90 = {np.percentile(sim_arr, 10):.2f}/"
        f"{np.percentile(sim_arr, 50):.2f}/{np.percentile(sim_arr, 90):.2f})")
    return pairs, names, quals, toks, sum(fam_pair_counts.values())


# --------------------------------------------------------------------------
# 4. band stratification
# --------------------------------------------------------------------------

def stratify(pairs, band, quals, target):
    by_band = defaultdict(list)
    for p in pairs:
        by_band[int(band[p[1]])].append(p)
    for b in by_band:
        by_band[b].sort(key=lambda p: md5x(quals[p[1]] + "|" + quals[p[2]]))
    # p = (fam_code, a, b, sim)

    alloc = {b: min(int(round(target * BAND_SHARE[b])), len(by_band.get(b, [])))
             for b in range(5)}
    # independent per-band rounding can overshoot the target (a target of 8
    # allocated 9): trim the excess from the largest bands
    while sum(alloc.values()) > target:
        big = max(alloc, key=alloc.get)
        alloc[big] -= 1
    scarce = target - sum(alloc.values())
    while scarce > 0:
        movable = [b for b in range(5) if alloc[b] < len(by_band.get(b, []))]
        if not movable:
            break
        for b in sorted(movable, key=lambda b: -len(by_band.get(b, []))):
            take = min(scarce, len(by_band[b]) - alloc[b])
            alloc[b] += take
            scarce -= take
            if scarce <= 0:
                break
    log(f"band allocation (supply -> alloc): "
        + ", ".join(f"b{b}:{len(by_band.get(b, []))}->{alloc[b]}"
                    for b in range(5)))
    return by_band, alloc


# --------------------------------------------------------------------------
# 5. filler pool (other-family stack members)
# --------------------------------------------------------------------------

def build_filler_pool(fam_codes, stub, corpus):
    rng = random.Random(f"{SEED}|i6-fillers")
    n = len(fam_codes)
    non_stub = np.flatnonzero(~stub)
    picks = rng.sample(range(len(non_stub)), min(FILLER_POOL_SIZE, len(non_stub)))
    idxs = sorted(int(non_stub[i]) for i in picks)
    log(f"filler candidates: {len(idxs)}; fetching name/qualified/code ...")
    names = corpus.fetch("func_name", idxs)
    quals = corpus.fetch("qualified", idxs)
    codes = corpus.fetch("code", idxs)
    pool, tok_sets = [], {}
    n_bad = 0
    for i in idxs:
        try:
            make_ctx({"code": codes[i], "qualified": quals[i]}, i)
        except (ParseFailure, SyntaxError, RecursionError, ValueError,
                MemoryError):
            n_bad += 1
            continue
        pool.append(i)
        tok_sets[i] = frozenset(content_tokens(codes[i], short(names[i])))
    log(f"filler pool: {len(pool)} parse-ok ({n_bad} rejected)")
    assert 0 <= min(pool) and max(pool) < n
    return pool, names, quals, tok_sets


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------
# 6. assembly + emission
# --------------------------------------------------------------------------

def assemble(by_band, alloc, fam_codes, repo, names, quals, toks, corpus,
             pool, pool_names, pool_quals, pool_toksets):
    """Match rows are parse-screened upstream (build_pairs)."""
    triples = []
    pair_sims = {}        # target row_idx -> pair sim (for examples/report)
    stats = Counter()
    taken_band = Counter()
    for b in range(5):
        for fc, a, bx, sim in by_band.get(b, []):
            if taken_band[b] >= alloc[b]:
                break
            rng = random.Random(int(md5x(quals[a] + "|" + quals[bx])[:12], 16))
            na, nb = short(names[a]), short(names[bx])
            target_toks = frozenset(toks[a]) | frozenset(toks[bx])
            # Filler names must also remain unique after lowercase grading.
            used_names = {na.casefold(), nb.casefold()}
            used_fams = {fc}
            fillers = []
            for _ in range(400):
                if len(fillers) == STACK_SIZE - 2:
                    break
                fi = pool[rng.randrange(len(pool))]
                ffc = int(fam_codes[fi])
                fname = short(pool_names[fi]).casefold()
                if (ffc in used_fams or fname in used_names
                        or fi in (a, bx)):
                    continue
                # a filler content-similar to the matches could genuinely
                # satisfy the description too -> would break "exactly two"
                if _jaccard(pool_toksets[fi], target_toks) >= FILLER_MAX_JACCARD:
                    stats["filler_near_match_rejected"] += 1
                    continue
                fillers.append(fi)
                used_fams.add(ffc)
                used_names.add(fname)
            if len(fillers) != STACK_SIZE - 2:
                stats["drop_fillers"] += 1
                continue
            qual_of = {a: quals[a], bx: quals[bx],
                       **{f: pool_quals[f] for f in fillers}}
            stack = sorted([a, bx] + fillers, key=lambda i: md5x(qual_of[i]))
            triples.append({
                "row_idx": a,
                "qualified": quals[a],
                "intent": "I6",
                "group": "I",
                "held_out": False,
                "needs_luna": False,
                "needs_phrasing": True,
                "question_seed": {
                    "intent": "I6",
                    "describe_row_idx": a,
                    "match_row_idxs": sorted([a, bx]),
                    "stack_row_idxs": stack,
                },
                "answer": ", ".join(sorted([na, nb])),
                "answer_type": "name",
            })
            pair_sims[a] = round(sim, 4)
            taken_band[b] += 1
    log(f"assembled {len(triples)} triples "
        f"(per band {dict(sorted(taken_band.items()))}; "
        f"drops/rejects {dict(stats)})")
    return triples, pair_sims


# --------------------------------------------------------------------------
# 7. verification + examples
# --------------------------------------------------------------------------

def verify(triples, fam_codes, repo, n_train, corpus, k):
    rng = random.Random(f"{SEED}|i6-verify")
    sample = rng.sample(triples, min(k, len(triples)))
    if not sample:
        log("verify: EMPTY supply — nothing to verify, failing closed")
        return False
    need = sorted({i for t in sample
                   for i in t["question_seed"]["stack_row_idxs"]})
    codes = corpus.fetch("code", need)
    names = corpus.fetch("func_name", need)

    n_ok = 0
    sims, csims = [], []
    for t in sample:
        seed = t["question_seed"]
        a, b = seed["match_row_idxs"]
        stack = seed["stack_row_idxs"]
        problems = []
        # in-range (corpus is post-exclusion; range check = the whole game)
        if not all(0 <= i < n_train for i in stack):
            problems.append("row_idx out of range")
        # skeleton family recompute (independent of annot)
        ha, _ = skeleton_hash(codes[a])
        hb, _ = skeleton_hash(codes[b])
        if ha != hb:
            problems.append("skeleton hash mismatch")
        if fam_codes[a] != fam_codes[b]:
            problems.append("fam code mismatch")
        sim = difflib.SequenceMatcher(None, codes[a], codes[b]).ratio()
        sims.append(sim)
        na, nb = short(names[a]), short(names[b])
        ta = content_tokens(codes[a], na)
        tb = content_tokens(codes[b], nb)
        csim = pair_sim(ta, tb)
        csims.append(csim)
        if csim < _floor_for(min(len(ta), len(tb))):
            problems.append(f"content sim {csim:.2f} below graded floor")
        if na.casefold() == nb.casefold():
            problems.append("names not distinct (casefold)")
        if repo[a] == repo[b]:
            problems.append("same repo")
        if t["answer"] != ", ".join(sorted([na, nb])):
            problems.append("answer format")
        if seed["describe_row_idx"] not in (a, b) or t["row_idx"] != \
                seed["describe_row_idx"]:
            problems.append("describe target inconsistent")
        fills = [i for i in stack if i not in (a, b)]
        ffams = [int(fam_codes[i]) for i in fills]
        if int(fam_codes[a]) in ffams:
            problems.append("filler from match family")
        if len(set(ffams)) != len(ffams):
            problems.append("filler families collide")
        snames = [short(names[i]).casefold() for i in stack]
        if len(set(snames)) != len(snames):
            problems.append("stack names not unique (casefold)")
        # leak rule: no string field of the seed may contain either name
        for v in seed.values():
            if isinstance(v, str) and (na in v or nb in v):
                problems.append("name leak in seed")
        if problems:
            print(f"  VERIFY FAIL row_idx {t['row_idx']}: {problems}")
        else:
            n_ok += 1
    log(f"verify: {n_ok}/{len(sample)} clean; raw char sim "
        f"min/mean/max {min(sims):.3f}/{sum(sims)/len(sims):.3f}/"
        f"{max(sims):.3f}; content-token sim min/mean/max "
        f"{min(csims):.3f}/{sum(csims)/len(csims):.3f}/{max(csims):.3f}")
    return n_ok == len(sample)


def dump_examples(triples, pair_sims, corpus, n=15):
    rng = random.Random(f"{SEED}|i6-examples")
    picks = rng.sample(triples, min(n, len(triples)))
    need = sorted({i for t in picks
                   for i in t["question_seed"]["stack_row_idxs"]})
    codes = corpus.fetch("code", need)
    names = corpus.fetch("func_name", need)
    repos = corpus.fetch("repo", need)
    OUT_EXAMPLES.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_EXAMPLES, "w") as f:
        for t in picks:
            a, b = t["question_seed"]["match_row_idxs"]
            rec = dict(t)
            rec["code"] = codes[a][:2000]
            rec["code_match_b"] = codes[b][:2000]
            rec["match_repos"] = [repos[a], repos[b]]
            rec["pair_content_sim"] = pair_sims.get(
                t["question_seed"]["describe_row_idx"])
            rec["sibling_names"] = [short(names[i]) for i in
                                    t["question_seed"]["stack_row_idxs"]]
            f.write(json.dumps(rec) + "\n")
    log(f"wrote {len(picks)} examples -> {OUT_EXAMPLES}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=12_000)
    ap.add_argument("--verify", type=int, default=30)
    args = ap.parse_args()
    if args.verify < 1:
        ap.error("--verify must be >= 1")

    fam_codes, repo, band, stub = load_kept_meta()
    corpus = Corpus()
    assert len(corpus) == len(fam_codes), "corpus/manifest length mismatch"

    pairs, names, quals, toks, n_families = build_pairs(
        fam_codes, repo, stub, corpus)
    by_band, alloc = stratify(pairs, band, quals, args.target)
    pool, pool_names, pool_quals, pool_toksets = build_filler_pool(
        fam_codes, stub, corpus)
    triples, pair_sims = assemble(by_band, alloc, fam_codes, repo, names,
                                  quals, toks, corpus, pool, pool_names,
                                  pool_quals, pool_toksets)

    # global invariants
    targets = [t["row_idx"] for t in triples]
    assert len(set(targets)) == len(targets), "duplicate target row_idx"
    for t in triples:
        s = t["question_seed"]
        assert len(s["stack_row_idxs"]) == STACK_SIZE
        assert set(s["match_row_idxs"]) <= set(s["stack_row_idxs"])
        assert all(0 <= i < len(corpus) for i in s["stack_row_idxs"])

    # Verify the temporary file before atomically publishing the canonical path.
    OUT_TRIPLES.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_TRIPLES.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for t in triples:
            f.write(json.dumps(t) + "\n")

    ok = verify(triples, fam_codes, repo, len(corpus), corpus, args.verify)
    dump_examples(triples, pair_sims, corpus)
    if ok:
        tmp.rename(OUT_TRIPLES)
        log(f"wrote {len(triples)} triples -> {OUT_TRIPLES}")
    else:
        tmp.unlink()
        log(f"verification FAILED — {OUT_TRIPLES} NOT published "
            f"(temp removed)")
    log(f"supply {len(triples)}/{args.target} from {n_families} families "
        f"— verification {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
