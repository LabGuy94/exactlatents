"""Corpus sampler stage C: materialize the drawn corpus and metadata.

Streams source piles in canonical order. Lines absent from the draw manifest
are skipped without JSON parsing because the line index is the key, keeping the
stage mostly I/O-bound. Emits:

- data/corpus_v2/corpus/            HF DatasetDict (train + validation),
  csn_python-compatible column head (repo, path, func_name, code, n_tokens)
  plus provenance columns the trainer ignores.
- corpus_meta/eyeball_150.jsonl     150 random final-corpus rows (full code)
- corpus_meta/surprisal_5k.jsonl    stratified 5k surprisal-review sample
- corpus_meta/junk_sample_20.jsonl  20 random junk-flagged rows (full code)
- corpus_meta/families_top50.json   updated in place with example snippets
- corpus_meta/roundtrip_rows.jsonl  the 1000-row span round-trip sample
                                       (code + span + parent pointers)

Run: uv run python dataset/materialize.py
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset"))
from annotate import BUILD_NOW, PILES, shard_tasks  # noqa: E402

CV2 = ROOT / "data" / "corpus_v2"
META = CV2 / "corpus_meta"
OUT = CV2 / "corpus"

t0 = time.time()
m = pd.read_parquet(META / "draw_manifest.parquet")
kept = m[m.status.isin(["kept_train", "kept_val"])]
# (pile, shard) -> {line: split}
want = defaultdict(dict)
for pile, shard, line, statusv in zip(kept.pile, kept.shard, kept.line, kept.status):
    want[(pile, shard)][line] = statusv
n_train = int((kept.status == "kept_train").sum())
n_val = int((kept.status == "kept_val").sum())
print(f"materializing {n_train} train + {n_val} val rows", flush=True)

# special row sets (full code allowed in these meta files)
def load_ids(name):
    ids = json.load(open(META / name))
    out = defaultdict(set)
    for s in ids:
        p, sh, ln = s.split(":")
        out[(int(p[1:]), int(sh[1:]))].add(int(ln[1:]))
    return out, set(ids)

eye_map, _ = load_ids("eyeball_ids.json")
surp_map, _ = load_ids("surprisal_ids.json")
junk_map, _ = load_ids("junk_sample_ids.json")
rt_map, _ = load_ids("roundtrip_ids.json")
fam50 = json.load(open(META / "families_top50.json"))
fam_map = defaultdict(dict)
for fi, fr in enumerate(fam50):
    p, sh, ln = fr["example_id"].split(":")
    fam_map[(int(p[1:]), int(sh[1:]))][int(ln[1:])] = fi

sink_eye = open(META / "eyeball_150.jsonl", "w")
sink_surp = open(META / "surprisal_5k.jsonl", "w")
sink_junk = open(META / "junk_sample_20.jsonl", "w")
sink_rt = open(META / "roundtrip_rows.jsonl", "w")
tmp_train = open(META / "_train_rows.jsonl", "w")
tmp_val = open(META / "_val_rows.jsonl", "w")


def final_row(r, pile):
    ts = r.get("file_ts")
    if ts is not None and ts > BUILD_NOW:
        ts = None
    return {
        "repo": r["repo"], "path": r["path"], "func_name": r["func_name"],
        "code": r["code"], "n_tokens": r["n_tokens"],
        "qualified": r["qualified"], "span": r["span"],
        "is_method": bool(r["is_method"]), "is_async": bool(r["is_async"]),
        "decorated": bool(r["decorated"]), "has_docstring": bool(r["has_docstring"]),
        "typed": bool(r["typed"]), "string_share": float(r["string_share"]),
        "n_strings": int(r["n_strings"]),
        "source": PILES[pile], "file_ts": ts,
        "commit_id": r.get("commit_id"), "repo_created": r.get("repo_created"),
    }


done = 0
for pile, shard_idx, path in shard_tasks():
    key = (pile, shard_idx)
    lines_want = want.get(key, {})
    extras = (eye_map.get(key, set()) | surp_map.get(key, set())
              | junk_map.get(key, set()) | rt_map.get(key, set())
              | set(fam_map.get(key, {})))
    if not lines_want and not extras:
        continue
    with open(path) as f:
        for ln, line in enumerate(f):
            split = lines_want.get(ln)
            if split is None and ln not in extras:
                continue
            r = json.loads(line)
            rid = f"p{pile}:s{shard_idx}:l{ln}"
            if split is not None:
                fr = final_row(r, pile)
                (tmp_train if split == "kept_train" else tmp_val).write(
                    json.dumps(fr) + "\n")
                done += 1
            if ln in eye_map.get(key, set()):
                sink_eye.write(json.dumps({"id": rid, "source": PILES[pile], **{
                    k: r[k] for k in ("repo", "path", "qualified", "code", "n_tokens")}}) + "\n")
            if ln in surp_map.get(key, set()):
                sink_surp.write(json.dumps({"id": rid, "source": PILES[pile],
                                            "repo": r["repo"], "path": r["path"],
                                            "func_name": r["func_name"],
                                            "code": r["code"],
                                            "n_tokens": r["n_tokens"]}) + "\n")
            if ln in junk_map.get(key, set()):
                sink_junk.write(json.dumps({"id": rid, "source": PILES[pile],
                                            "repo": r["repo"], "code": r["code"],
                                            "n_tokens": r["n_tokens"]}) + "\n")
            if ln in rt_map.get(key, set()):
                sink_rt.write(json.dumps({"id": rid, "pile": pile,
                                          "repo": r["repo"], "path": r["path"],
                                          "span": r["span"], "code": r["code"]}) + "\n")
            if ln in fam_map.get(key, {}):
                fam50[fam_map[key][ln]]["example_snippet"] = r["code"][:300]
    if done and done % 200000 < 5000:
        print(f"  {done}/{n_train+n_val} rows, {(time.time()-t0)/60:.1f}m", flush=True)

for s in (sink_eye, sink_surp, sink_junk, sink_rt, tmp_train, tmp_val):
    s.close()
json.dump(fam50, open(META / "families_top50.json", "w"), indent=1)
print(f"streamed {done} rows in {(time.time()-t0)/60:.1f}m; building HF dataset ...",
      flush=True)

from datasets import Dataset, DatasetDict, Features, Sequence, Value

features = Features({
    "repo": Value("string"), "path": Value("string"), "func_name": Value("string"),
    "code": Value("string"), "n_tokens": Value("int64"),
    "qualified": Value("string"), "span": Sequence(Value("int64"), length=2),
    "is_method": Value("bool"), "is_async": Value("bool"),
    "decorated": Value("bool"), "has_docstring": Value("bool"),
    "typed": Value("bool"), "string_share": Value("float64"),
    "n_strings": Value("int32"),
    "source": Value("string"), "file_ts": Value("int64"),
    "commit_id": Value("string"), "repo_created": Value("string"),
})


def gen(path):
    def g():
        for line in open(path):
            yield json.loads(line)
    return g


cache = str(META / "_hfcache")
train_ds = Dataset.from_generator(gen(META / "_train_rows.jsonl"),
                                  features=features, cache_dir=cache)
val_ds = Dataset.from_generator(gen(META / "_val_rows.jsonl"),
                                features=features, cache_dir=cache)
dd = DatasetDict({"train": train_ds, "validation": val_ds})
dd.save_to_disk(str(OUT))
print(dd)
print(f"saved {OUT} in {(time.time()-t0)/60:.1f}m total", flush=True)
