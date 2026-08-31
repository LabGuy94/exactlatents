"""Reconstruct the released OOD-600 function battery.

The published pf4 receipt contains the exact function source used by the
evaluation and is the default input. Each function is compressed, greedily
reconstructed, and scored with the repository's exactness metrics.

Run from the repository root:
    python eval/ood_battery.py
"""

import argparse
import difflib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from compressor import Compressor, ft_load
from compressor.exactness import byte_exact, code_exact
from compressor.genutil import batched_generate

ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("--pf", type=int, default=4)
p.add_argument("--decoder", type=str, default="Qwen/Qwen3-1.7B")
p.add_argument("--encoder", type=str, default="Qwen/Qwen3-1.7B")
p.add_argument("--proj-width", type=int, default=None)
p.add_argument("--proj-depth", type=int, default=2)
p.add_argument("--pooling", type=str, default="latent", choices=["mean", "latent"],
               help="Must match the released checkpoint.")
p.add_argument("--boundary", action=argparse.BooleanOptionalAction, default=True)
p.add_argument("--bidir-encoder", action="store_true")
p.add_argument("--projector-weights", type=Path,
               default=ROOT / "weights" / "projector_ema.safetensors",
               help="EMA projector weights.")
p.add_argument("--model-weights", type=Path,
               default=ROOT / "weights" / "model.safetensors",
               help="Encoder and decoder weights.")
p.add_argument("--items", type=Path,
               help="JSONL containing func_name and code for each OOD function. "
                    "Defaults to the published receipt for --pf.")
p.add_argument("--expected-n", type=int, default=600,
               help="Expected number of unique functions; 0 disables the check.")
p.add_argument("--limit", type=int, default=None,
               help="Evaluate only the first N functions.")
p.add_argument("--gen-batch", type=int, default=8)
p.add_argument("--out", type=Path,
               help="Per-function JSONL receipt. Defaults to "
                    "results/ood_battery_pf<pf>.jsonl.")
args = p.parse_args()
if args.items is None:
    args.items = (
        ROOT / "receipts" / "ood600" / f"ood600_pf{args.pf}_gens.jsonl"
    )

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

with args.items.open(encoding="utf-8") as source:
    items = [json.loads(line) for line in source if line.strip()]
assert items, f"{args.items}: no items"
# The released receipt preserves one duplicated first record from collection.
if len(items) > 1 and all(
    items[0].get(key) == items[1].get(key) for key in ("func_name", "code")
):
    del items[1]
for key in ("func_name", "code"):
    assert key in items[0], f"{args.items}: record missing {key!r}"
if args.limit:
    items = items[: args.limit]
    print(f"limited evaluation: {len(items)} functions")
elif args.expected_n:
    assert len(items) == args.expected_n, (
        f"battery has {len(items)} unique items, expected {args.expected_n}"
    )

dec_hidden = AutoConfig.from_pretrained(args.decoder).hidden_size
comp = Compressor(encoder_name=args.encoder, decoder_hidden=dec_hidden,
                  pooling_factor=args.pf, proj_width=args.proj_width,
                  proj_depth=args.proj_depth, pooling=args.pooling,
                  boundary=args.boundary, bidirectional=args.bidir_encoder).to(device)
load_meta = {
    "projector": ft_load.load_projector(
        comp.projector, args.projector_weights, prefer="ema"
    ),
    "encoder": ft_load.load_encoder(comp, args.model_weights),
}
comp.eval()

tok = AutoTokenizer.from_pretrained(args.decoder)
decoder = AutoModelForCausalLM.from_pretrained(args.decoder, dtype=torch.bfloat16).to(device)
load_meta["decoder"] = ft_load.load_decoder(decoder, args.model_weights)
decoder.eval()

out_path = args.out or ROOT / "results" / f"ood_battery_pf{args.pf}.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
print(f"battery: {len(items)} items ({args.items}) | gen-batch {args.gen_batch} "
      f"| out {out_path}")
print(f"weights: {json.dumps(load_meta, sort_keys=True)}")

all_true_ids = [tok(it["code"], add_special_tokens=False).input_ids for it in items]
for item, token_ids in zip(items, all_true_ids):
    item.setdefault("n_tok", len(token_ids))
with torch.no_grad():
    vec_list = [comp.compress(it["code"], device)[0] for it in items]
outs = batched_generate(decoder, vec_list, [len(t) + 32 for t in all_true_ids],
                        batch_size=args.gen_batch)

results, n_empty = [], 0
with open(out_path, "w") as f:
    for it, out in zip(items, outs):
        code = it["code"]
        gen = tok.decode(out.tolist(), skip_special_tokens=True)
        n_empty += not gen.strip()
        sim = difflib.SequenceMatcher(None, code, gen[: len(code) + 200],
                                      autojunk=False).ratio()
        r = {
            "func_name": it["func_name"],
            "code": code,
            "n_tok": it["n_tok"],
            "repo": it.get("repo"),
            "permalink": it.get("permalink"),
            "is_method": it.get("is_method"),
            "decorated": it.get("decorated"),
            "is_async": it.get("is_async"),
            "has_docstring": it.get("has_docstring"),
            "string_share": it.get("string_share"),
            "byte_exact": byte_exact(code, gen),
            "code_exact": code_exact(code, gen),
            "sim": round(100 * sim, 1),
            "gen": gen,
        }
        results.append(r)
        f.write(json.dumps(r) + "\n")
        tag = ("BYTE " if r["byte_exact"] else "CODE " if r["code_exact"]
               else f"{r['sim']:5.1f}")
        print(f"{tag}  ({it['n_tok']:3d} tok)  {it['func_name']}")
    f.write(json.dumps({"_meta": {"argv": sys.argv, "loads": load_meta,
                                  "n_items": len(results)}}) + "\n")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def line(label, rs):
    if not rs:
        return f"  {label:>22}: (none)"
    nb = sum(r["byte_exact"] for r in rs)
    nc = sum(r["code_exact"] for r in rs)
    sm = sum(r["sim"] for r in rs) / len(rs)
    return (f"  {label:>22}: {nc}/{len(rs)} code ({100*nc/len(rs):.1f}%) | "
            f"{nb} byte | sim {sm:.1f}")


n = len(results)
nc = sum(r["code_exact"] for r in results)
nb = sum(r["byte_exact"] for r in results)
lo, hi = wilson(nc, n)
print(f"\nSUMMARY: {nc}/{n} code-exact = {100*nc/n:.1f}% "
      f"(95% CI {100*lo:.1f}-{100*hi:.1f}) | {nb}/{n} byte-exact | "
      f"avg sim {sum(r['sim'] for r in results)/n:.1f}%")
if n_empty:
    print(f"!! {n_empty} EMPTY generations — decoder produced nothing !!")
print("by length:")
for a, b in ((50, 150), (150, 300), (300, 450), (450, 600)):
    print(line(f"{a}-{b} tok", [r for r in results if a <= r["n_tok"] < b or
                                (b == 600 and r["n_tok"] == 600)]))
print("by type:")
for label, pred in (
    ("method", lambda r: r["is_method"] is True),
    ("free function", lambda r: r["is_method"] is False),
    ("async", lambda r: r["is_async"] is True),
    ("decorated", lambda r: r["decorated"] is True),
    ("with docstring", lambda r: r["has_docstring"] is True),
    ("string-free", lambda r: r["string_share"] == 0),
    ("string-heavy >15%", lambda r: r["string_share"] is not None and r["string_share"] > 0.15),
    ("string-drenched >30%", lambda r: r["string_share"] is not None and r["string_share"] > 0.30),
):
    print(line(label, [r for r in results if pred(r)]))

if args.limit:
    print("\n(limited evaluation)")
print(f"receipt: {out_path}")
