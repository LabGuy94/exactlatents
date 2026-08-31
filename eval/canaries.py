"""Reconstruct the released canary set from continuous latent vectors.

Each function is compressed, greedily reconstructed, and scored for byte and
code exactness. Optional diffs make non-exact generations easy to inspect.

Run from the repository root:
    python eval/canaries.py
"""

import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from compressor import Compressor

ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("--pf", type=int, default=4)
p.add_argument("--decoder", type=str, default="Qwen/Qwen3-1.7B")
p.add_argument("--encoder", type=str, default="Qwen/Qwen3-1.7B")
p.add_argument("--proj-width", type=int, default=None)
p.add_argument("--proj-depth", type=int, default=2)
p.add_argument("--pooling", type=str, default="latent", choices=["mean", "latent"],
               help="Must match the released checkpoint.")
p.add_argument("--projector-weights", type=Path,
               default=ROOT / "weights" / "projector_ema.safetensors",
               help="EMA projector weights.")
p.add_argument("--model-weights", type=Path,
               default=ROOT / "weights" / "model.safetensors",
               help="Encoder and decoder weights.")
p.add_argument("--boundary", action=argparse.BooleanOptionalAction, default=True)
p.add_argument("--show-diffs", type=int, default=3)
p.add_argument("--beams", type=int, default=1,
               help="Beam width for generation; 1 uses greedy decoding.")
p.add_argument("--save-gens", type=Path,
               default=ROOT / "results" / "canaries.jsonl",
               help="Per-function JSONL receipt.")
p.add_argument("--gen-batch", type=int, default=1,
               help="Functions generated per batch. Serial generation is the "
                    "most numerically stable setting.")
p.add_argument("--canaries", type=Path,
               default=ROOT / "data" / "canaries_v2.jsonl",
               help="Canary JSONL file.")
p.add_argument("--bidir-encoder", action="store_true",
               help="Allow the encoder to attend bidirectionally.")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

dec_hidden = AutoConfig.from_pretrained(args.decoder).hidden_size
comp = Compressor(encoder_name=args.encoder, decoder_hidden=dec_hidden, pooling_factor=args.pf,
                  proj_width=args.proj_width, proj_depth=args.proj_depth,
                  pooling=args.pooling, boundary=args.boundary, bidirectional=args.bidir_encoder).to(device)
from compressor import ft_load

load_meta = {
    "projector": ft_load.load_projector(
        comp.projector, args.projector_weights, prefer="ema"
    ),
    "encoder": ft_load.load_encoder(comp, args.model_weights),
}
comp.eval()

tok = AutoTokenizer.from_pretrained(args.decoder)
decoder = AutoModelForCausalLM.from_pretrained(
    args.decoder, dtype=torch.bfloat16
).to(device)
load_meta["decoder"] = ft_load.load_decoder(decoder, args.model_weights)
decoder.eval()

canaries = [json.loads(line) for line in args.canaries.open(encoding="utf-8")]
print(f"canaries: {len(canaries)} ({args.canaries})")
print(f"weights: {json.dumps(load_meta, sort_keys=True)}\n")

from compressor.genutil import batched_generate

all_true_ids = [tok(c["code"], add_special_tokens=False).input_ids for c in canaries]
with torch.no_grad():
    vec_list = [comp.compress(c["code"], device)[0] for c in canaries]
outs = batched_generate(decoder, vec_list, [len(t) + 32 for t in all_true_ids],
                        batch_size=args.gen_batch, num_beams=args.beams)

results = []
for c, true_ids, out in zip(canaries, all_true_ids, outs):
    code = c["code"]
    gen_ids = out.tolist()
    gen = tok.decode(gen_ids, skip_special_tokens=True)

    n = min(len(true_ids), len(gen_ids))
    tok_acc = sum(a == b for a, b in zip(true_ids[:n], gen_ids[:n])) / max(len(true_ids), 1)
    # Positional accuracy is sensitive to a single early insertion. Aligned
    # token F1 remains informative when an otherwise faithful generation shifts.
    sm_t = difflib.SequenceMatcher(
        None, true_ids, gen_ids[: len(true_ids) + 64], autojunk=False
    )
    matched = sum(block.size for block in sm_t.get_matching_blocks())
    tok_f1 = 2 * matched / max(
        len(true_ids) + min(len(gen_ids), len(true_ids) + 64), 1
    )
    # Disable difflib's long-string frequency heuristic for source code.
    sim = difflib.SequenceMatcher(
        None, code, gen[: len(code) + 200], autojunk=False
    ).ratio()
    exact = gen.startswith(code.strip()) or code.strip() == gen.strip()[: len(code.strip())]
    from compressor.exactness import byte_exact, code_exact
    b_exact = byte_exact(code, gen)
    c_exact = code_exact(code, gen)
    results.append({"name": c["func_name"], "n_tokens": c["n_tokens"],
                    "tier": c.get("tier", "v1"),
                    "exact": exact, "byte_exact": b_exact, "code_exact": c_exact,
                    "tok_acc": tok_acc, "tok_f1": tok_f1, "sim": sim,
                    "code": code, "gen": gen})
    tag = "BYTE  " if b_exact else "CODE  " if c_exact else f"{100*sim:5.1f}%"
    print(f"{tag}  tok_f1 {100*tok_f1:5.1f}%  "
          f"({c['n_tokens']:3d} tok)  {c['func_name']}")
if args.save_gens:
    args.save_gens.parent.mkdir(parents=True, exist_ok=True)
    with args.save_gens.open("w", encoding="utf-8") as receipt:
        for result in results:
            receipt.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"receipt: {args.save_gens}")

n_exact = sum(r["exact"] for r in results)
n_byte = sum(r["byte_exact"] for r in results)
n_code = sum(r["code_exact"] for r in results)
avg_sim = sum(r["sim"] for r in results) / len(results)
avg_acc = sum(r["tok_acc"] for r in results) / len(results)
print(f"\nSUMMARY: {n_byte}/{len(results)} byte-exact | {n_code}/{len(results)} code-exact "
      f"| (legacy exact {n_exact}) | avg similarity {100*avg_sim:.1f}% "
      f"| avg token accuracy {100*avg_acc:.1f}% "
      f"| avg token F1 {100*sum(r['tok_f1'] for r in results)/len(results):.1f}%")
tiers = sorted({r["tier"] for r in results})
if tiers != ["v1"]:
    # Tier lines retain both exactness metrics for each canary category.
    for t in tiers:
        rs = [r for r in results if r["tier"] == t]
        print(f"  [{t:>10}] {sum(r['byte_exact'] for r in rs)}/{len(rs)} byte | "
              f"{sum(r['code_exact'] for r in rs)}/{len(rs)} code | "
              f"sim {100*sum(r['sim'] for r in rs)/len(rs):.1f}% | "
              f"tok_acc {100*sum(r['tok_acc'] for r in rs)/len(rs):.1f}%")

print("\n" + "=" * 70)
for r in sorted(results, key=lambda r: r["sim"])[: args.show_diffs]:
    print(f"\n--- WORST-{args.show_diffs} DIFF: {r['name']} (sim {100*r['sim']:.1f}%) ---")
    diff = difflib.unified_diff(
        r["code"].splitlines(keepends=True),
        r["gen"][: len(r["code"]) + 200].splitlines(keepends=True),
        fromfile="original", tofile="reconstructed", n=1,
    )
    sys.stdout.writelines(list(diff)[:40])
