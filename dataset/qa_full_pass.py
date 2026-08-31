"""Run AST-only QA extraction over every training-corpus row.

This streaming variant of ``qa_coverage.py`` writes triples to
``triples_ast_XXXX.jsonl`` shards instead of retaining them in memory. It also
writes ``exec_manifest.jsonl`` for every function that passes the
self-containedness screen and ``full_pass_stats.json`` for verification.

The driver reuses the QA registry, execution screen, and pool/stack builders.
Its distractor pool is a deterministic shuffled sample of docstring rows across
the full corpus, avoiding repository-order bias.

Usage:
  uv run python dataset/qa_full_pass.py --workers 14
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore", category=SyntaxWarning)

import qa_exec
import qa_extract
from qa_extract import ParseFailure, extract_row, fact_to_triple, make_ctx
import qa_coverage
from qa_coverage import (SAMPLE_SEED, _init_worker, _pool_worker, build_stacks,
                         extract_stacks)

SHARD_SIZE = 500_000


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard = -1
        self.in_shard = SHARD_SIZE  # force open on first write
        self.total = 0
        self.f = None

    def _roll(self):
        if self.f:
            self.f.close()
        self.shard += 1
        self.in_shard = 0
        self.f = open(self.out_dir / f"{self.prefix}_{self.shard:04d}.jsonl", "w")

    def write(self, triple: dict):
        if self.in_shard >= SHARD_SIZE:
            self._roll()
        self.f.write(json.dumps(triple) + "\n")
        self.in_shard += 1
        self.total += 1

    def close(self):
        if self.f:
            self.f.close()


def _extract_worker_ast(chunk):
    """chunk: [(row_idx, row)]. AST-only: like qa_coverage._extract_worker but
    returns (row_idx, qualified) manifest entries instead of exec combos."""
    triples, manifest, ok = [], [], []
    stats = {
        "rows": 0, "parse_fail": 0, "normalized": 0, "leak_dropped": 0,
        "applicable": Counter(), "emitted": Counter(),
        "screen_pass": 0, "screen_reasons": Counter(),
    }
    for row_idx, row in chunk:
        stats["rows"] += 1
        try:
            ctx = make_ctx(row, row_idx, pools=qa_coverage._POOLS)
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
            manifest.append((row_idx, row["qualified"]))
        else:
            stats["screen_reasons"][s_reason.split(":")[0]] += 1
        facts, fstats = extract_row(ctx)
        stats["leak_dropped"] += fstats["leak_dropped"]
        for iid in fstats["applicable"]:
            stats["applicable"][iid] += 1
        for f in facts:
            stats["emitted"][f.intent] += 1
            triples.append(fact_to_triple(f, row, row_idx))
    return triples, stats, manifest, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_v2/corpus")
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out-dir", default="data/corpus_v2/qa_full")
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    args = ap.parse_args()

    from datasets import load_from_disk

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ds = load_from_disk(args.corpus)[args.split]
    n = len(ds)
    idxs = list(range(min(args.limit, n) if args.limit else n))
    keep = ["repo", "path", "func_name", "code", "n_tokens", "qualified",
            "is_method", "is_async", "decorated", "has_docstring", "typed",
            "string_share", "n_strings"]
    tbl = ds.data.table if hasattr(ds.data, "table") else ds.data
    cols = {k: tbl.column(k) for k in keep}
    rows = [{k: cols[k][i].as_py() for k in keep} for i in idxs]
    rows_by_idx = dict(zip(idxs, rows))
    timings = {"load": time.perf_counter() - t0}
    print(f"[load] {len(rows)} rows in {timings['load']:.1f}s", flush=True)

    # pools: seeded-shuffle draw of 30k docstring rows across the full corpus
    t0 = time.perf_counter()
    doc_idx = [i for i, r in enumerate(rows) if r["has_docstring"]]
    random.Random(SAMPLE_SEED).shuffle(doc_idx)
    doc_rows = [rows[i] for i in sorted(doc_idx[:30000])]
    pools = qa_coverage.build_pools(doc_rows, args.workers)
    timings["pools"] = time.perf_counter() - t0
    print(f"[pools] {len(pools['doc_first_lines'])} docstring lines "
          f"in {timings['pools']:.1f}s", flush=True)

    t0 = time.perf_counter()
    stats = {
        "rows": 0, "parse_fail": 0, "normalized": 0, "leak_dropped": 0,
        "applicable": Counter(), "emitted": Counter(), "screen_pass": 0,
        "screen_reasons": Counter(),
    }
    writer = ShardWriter(out_dir, "triples_ast")
    manifest_all, ok_idxs = [], []
    pairs = list(zip(idxs, rows))
    chunks = [pairs[i:i + 250] for i in range(0, len(pairs), 250)]
    n_chunks = len(chunks)
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(pools,)) as p:
        for ci, (tr, st, mf, ok) in enumerate(p.imap(_extract_worker_ast, chunks)):
            for t in tr:
                writer.write(t)
            manifest_all.extend(mf)
            ok_idxs.extend(ok)
            for k in ("rows", "parse_fail", "normalized", "leak_dropped",
                      "screen_pass"):
                stats[k] += st[k]
            stats["applicable"].update(st["applicable"])
            stats["emitted"].update(st["emitted"])
            stats["screen_reasons"].update(st["screen_reasons"])
            if (ci + 1) % 400 == 0 or ci + 1 == n_chunks:
                el = time.perf_counter() - t0
                done_rows = stats["rows"]
                eta = el / max(done_rows, 1) * (len(rows) - done_rows)
                print(f"[extract] {done_rows}/{len(rows)} rows | "
                      f"{writer.total} triples | {done_rows/el:,.0f} rows/s | "
                      f"ETA {eta/60:.1f} min", flush=True)
    timings["extract"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stacks = build_stacks(rows_by_idx, ok_idxs)
    stack_triples = extract_stacks(stacks, rows_by_idx, stats)
    for t in stack_triples:
        writer.write(t)
    writer.close()
    timings["stacks"] = time.perf_counter() - t0
    print(f"[stacks] {len(stacks)} stacks, {len(stack_triples)} triples "
          f"in {timings['stacks']:.1f}s", flush=True)

    with open(out_dir / "exec_manifest.jsonl", "w") as f:
        for row_idx, qualified in manifest_all:
            f.write(json.dumps({"row_idx": row_idx, "qualified": qualified}) + "\n")
    print(f"[manifest] {len(manifest_all)} screen-pass rows -> exec_manifest.jsonl",
          flush=True)

    stats_out = {
        "rows": stats["rows"], "parse_fail": stats["parse_fail"],
        "normalized": stats["normalized"], "leak_dropped": stats["leak_dropped"],
        "screen_pass": stats["screen_pass"],
        "screen_reasons": dict(stats["screen_reasons"]),
        "applicable": dict(stats["applicable"]),
        "emitted": dict(stats["emitted"]),
        "n_stacks": len(stacks), "n_triples": writer.total,
        "n_shards": writer.shard + 1, "manifest": len(manifest_all),
        "pool_lines": len(pools["doc_first_lines"]),
        "timings": {k: round(v, 1) for k, v in timings.items()},
    }
    with open(out_dir / "full_pass_stats.json", "w") as f:
        json.dump(stats_out, f, indent=1)
    print(f"[done] {writer.total} triples in {writer.shard + 1} shards -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
