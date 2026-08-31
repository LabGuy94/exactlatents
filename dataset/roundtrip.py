"""Check byte-exact span round trips against stored parent files.

Takes the 1000-row sample staged by materialize.py (roundtrip_rows.jsonl),
finds each row's stored parent file (starcoder/starcoder_long files parquets,
stackv3 files_*.jsonl.gz), slices bytes[span[0]:span[1]] from the parent
content and byte-compares (after the harvest's .strip(), which is how the code
column was produced) against the corpus row's code.

A (repo, path) key can match multiple stored parents (same file harvested at
different revisions); the row passes if ANY candidate slice matches, and the
ambiguity count is reported.

Run: uv run python dataset/roundtrip.py
"""
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
CV2 = ROOT / "data" / "corpus_v2"
META = CV2 / "corpus_meta"

rows = [json.loads(l) for l in open(META / "roundtrip_rows.jsonl")]
need = defaultdict(list)          # (pile, repo, path) -> [row, ...]
for r in rows:
    need[(r["pile"], r["repo"], r["path"])].append(r)
print(f"{len(rows)} rows, {len(need)} distinct parent files", flush=True)

# candidate parent contents per key
cands = defaultdict(list)
t0 = time.time()

for pile, d in ((1, CV2 / "starcoder" / "files"), (2, CV2 / "starcoder_long" / "files"),
                (3, CV2 / "starcoder_valley" / "files")):
    keys = {(r, p) for (pl, r, p) in need if pl == pile}
    for shard in sorted(d.glob("shard_*.parquet")):
        t = pq.read_table(shard, columns=["repo", "path"])
        rp = list(zip(t["repo"].to_pylist(), t["path"].to_pylist()))
        idx = [i for i, k in enumerate(rp) if k in keys]
        if idx:
            tc = pq.read_table(shard, columns=["repo", "path", "content"])
            for i in idx:
                cands[(pile, rp[i][0], rp[i][1])].append(tc["content"][i].as_py())
    print(f"pile {pile} scanned in {time.time()-t0:.0f}s", flush=True)

sv3_keys = {(r, p) for (pl, r, p) in need if pl == 0}
for w in sorted((CV2 / "stackv3_raw" / "pod_out").glob("w*")):
    for gz in sorted(w.glob("files_*.jsonl.gz")):
        with gzip.open(gz, "rt") as f:
            for line in f:
                fr = json.loads(line)
                if (fr["repo"], fr["path"]) in sv3_keys:
                    cands[(0, fr["repo"], fr["path"])].append(fr["content"])
print(f"stackv3 scanned in {time.time()-t0:.0f}s", flush=True)

ok = fail = no_parent = ambiguous = 0
failures = []
for r in rows:
    key = (r["pile"], r["repo"], r["path"])
    cs = cands.get(key, [])
    if not cs:
        no_parent += 1
        failures.append({"id": r["id"], "why": "parent file not found", **{k: r[k] for k in ("repo", "path")}})
        continue
    if len(cs) > 1:
        ambiguous += 1
    a, z = r["span"]
    hit = any(c.encode()[a:z].decode(errors="replace").strip() == r["code"] for c in cs)
    if hit:
        ok += 1
    else:
        fail += 1
        failures.append({"id": r["id"], "why": "slice mismatch", "n_candidates": len(cs),
                         **{k: r[k] for k in ("repo", "path", "span")}})

res = {"checked": len(rows), "ok": ok, "slice_mismatch": fail,
       "parent_missing": no_parent, "multi_parent_keys": ambiguous,
       "failures": failures}
json.dump(res, open(META / "roundtrip_result.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "failures"}, indent=1))
if failures:
    print("FAILURES (first 10):")
    for f in failures[:10]:
        print(" ", f)
