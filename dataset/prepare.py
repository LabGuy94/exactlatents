"""Prepare CodeSearchNet-Python for compressor training.

Steps:
  1. Download the python config (official train/valid/test splits — CSN split
     by repository, so held-out means held-out repos; verified below).
  2. Deduplicate: exact-hash within splits, and drop train rows whose code
     also appears in valid/test (leakage removal — eval copies win).
  3. Filter to 64..512 encoder tokens (Qwen3 tokenizer) — too short teaches
     nothing, too long exceeds the chunk size we target first.
  4. Save to data/csn_python plus 20 canary functions for qualitative evals.

Output columns: repo, path, func_name, code.
"""

import hashlib
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "csn_python"
ENCODER = "Qwen/Qwen3-1.7B"
MIN_TOK, MAX_TOK = 64, 512

KEEP = {
    "repository_name": "repo",
    "func_path_in_repository": "path",
    "func_name": "func_name",
    "whole_func_string": "code",
}


def code_hash(s: str) -> str:
    return hashlib.sha1(s.strip().encode()).hexdigest()


def main() -> None:
    print("downloading CodeSearchNet python ...", flush=True)
    raw = load_dataset("code-search-net/code_search_net", "python")
    raw = DatasetDict(
        {
            split: ds.remove_columns([c for c in ds.column_names if c not in KEEP]).rename_columns(KEEP)
            for split, ds in raw.items()
        }
    )
    for split, ds in raw.items():
        print(f"  raw {split}: {len(ds)}")

    # --- repo-disjointness check (CSN's own guarantee; verify, don't trust) ---
    repos = {split: set(ds["repo"]) for split, ds in raw.items()}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        inter = repos[a] & repos[b]
        print(f"  repo overlap {a}/{b}: {len(inter)}")
        assert not inter, f"split contamination: {list(inter)[:5]}"

    # --- dedup ---------------------------------------------------------------
    eval_hashes: set[str] = set()
    seen: set[str] = set()

    def dedup_split(ds, split):
        keep_rows = []
        for i, code in enumerate(ds["code"]):
            h = code_hash(code)
            if h in seen:
                continue
            if split == "train" and h in eval_hashes:
                continue  # exact code also in valid/test -> keep only the eval copy
            seen.add(h)
            if split != "train":
                eval_hashes.add(h)
            keep_rows.append(i)
        return ds.select(keep_rows)

    deduped = {}
    for split in ["validation", "test", "train"]:  # eval splits first
        seen -= eval_hashes  # allow eval hashes to block train, not each other twice
        before = len(raw[split])
        deduped[split] = dedup_split(raw[split], split)
        print(f"  dedup {split}: {before} -> {len(deduped[split])}")

    # --- token-length filter ---------------------------------------------------
    tok = AutoTokenizer.from_pretrained(ENCODER)

    def add_len(batch):
        return {"n_tokens": [len(x) for x in tok(batch["code"]).input_ids]}

    final = DatasetDict()
    for split, ds in deduped.items():
        ds = ds.map(add_len, batched=True, batch_size=1000, desc=f"tokenize {split}")
        before = len(ds)
        ds = ds.filter(lambda x: MIN_TOK <= x["n_tokens"] <= MAX_TOK)
        final[split] = ds
        print(f"  length filter {split}: {before} -> {len(ds)}")

    # --- save -------------------------------------------------------------------
    OUT.parent.mkdir(exist_ok=True)
    final.save_to_disk(str(OUT))

    canaries = final["test"].shuffle(seed=42).select(range(20))
    with open(OUT.parent / "canaries.jsonl", "w") as f:
        for row in canaries:
            f.write(json.dumps({k: row[k] for k in ["repo", "func_name", "code", "n_tokens"]}) + "\n")

    import statistics

    lens = final["train"]["n_tokens"]
    print("\nfinal dataset:")
    for split, ds in final.items():
        print(f"  {split}: {len(ds)} functions")
    print(f"  train tokens: total {sum(lens)/1e6:.1f}M, "
          f"mean {statistics.mean(lens):.0f}, median {statistics.median(lens):.0f}")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
