"""Measure interference from neighboring function latents.

Each input row describes one repository group with a ``functions`` list and
``target_idx`` positions. Targets are reconstructed alone, among same-repository
siblings, and among mixed-repository fillers.

Run from the repository root:
    python eval/stacks.py --stacks /path/to/eval_stacks.jsonl
"""
import argparse
import difflib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from compressor import Compressor
from compressor.exactness import byte_exact, code_exact

ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("--encoder", type=str, default="Qwen/Qwen3-1.7B")
p.add_argument("--decoder", type=str, default="Qwen/Qwen3-1.7B")
p.add_argument("--pf", type=int, default=4)
p.add_argument("--pooling", type=str, default="latent")
p.add_argument("--boundary", action=argparse.BooleanOptionalAction, default=True)
p.add_argument("--proj-width", type=int, default=None)
p.add_argument("--proj-depth", type=int, default=2)
p.add_argument("--projector-weights", type=Path,
               default=ROOT / "weights" / "projector_ema.safetensors",
               help="EMA projector weights.")
p.add_argument("--model-weights", type=Path,
               default=ROOT / "weights" / "model.safetensors",
               help="Encoder and decoder weights.")
p.add_argument("--stacks", type=Path, required=True,
               help="JSONL of repository groups used to construct stack contexts.")
p.add_argument("--out", type=Path,
               default=ROOT / "results" / "stacks.jsonl",
               help="Per-target JSONL receipt.")
p.add_argument("--max-new-frac", type=float, default=1.3)
args = p.parse_args()

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
dec_hidden = AutoConfig.from_pretrained(args.decoder).hidden_size
comp = Compressor(
    encoder_name=args.encoder,
    decoder_hidden=dec_hidden,
    pooling_factor=args.pf,
    proj_width=args.proj_width,
    proj_depth=args.proj_depth,
    pooling=args.pooling,
    boundary=args.boundary,
).to(device)
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
emb_t = decoder.get_input_embeddings()

def embeds(text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    return emb_t(ids)

with args.stacks.open(encoding="utf-8") as source:
    groups = [json.loads(line) for line in source if line.strip()]
rng = random.Random(7)
# mixed-repo filler pool: functions from all OTHER groups
allfns = [(gi, f) for gi, g in enumerate(groups) for f in g["functions"]]

@torch.no_grad()
def gen_from(ctx, true_code):
    true_ids = tok(true_code, add_special_tokens=False)["input_ids"]
    out = decoder.generate(inputs_embeds=ctx, max_new_tokens=int(len(true_ids) * args.max_new_frac) + 16,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0], skip_special_tokens=True)
    sim = difflib.SequenceMatcher(None, true_code, gen[:len(true_code) + 200], autojunk=False).ratio()
    return gen, {"byte": byte_exact(true_code, gen), "code": code_exact(true_code, gen),
                 "sim": round(100 * sim, 1)}

@torch.no_grad()
def build_ctx(fns, target_pos, target_code):
    parts = []
    for i, f in enumerate(fns):
        code = target_code if i == target_pos else f["code"]
        name = f["func_name"]
        parts.append(embeds(f"### function: {name}\n"))
        parts.append(comp.compress(code, device).to(emb_t.weight.dtype))
    tgt_name = fns[target_pos]["func_name"]
    parts.append(embeds(f"\nReproduce the function `{tgt_name}`:\n"))
    return torch.cat(parts, dim=1)

args.out.parent.mkdir(parents=True, exist_ok=True)
out_f = args.out.open("w", encoding="utf-8")
for gi, g in enumerate(groups):
    fns = g["functions"]
    for ti in g["target_idx"]:
        tgt = fns[ti]
        code = tgt["code"]
        # solo baseline
        solo_ctx = torch.cat([embeds(f"### function: {tgt['func_name']}\n"),
                              comp.compress(code, device).to(emb_t.weight.dtype),
                              embeds(f"\nReproduce the function `{tgt['func_name']}`:\n")], dim=1)
        gen_s, sc_s = gen_from(solo_ctx, code)
        # same-repo stack
        gen_r, sc_r = gen_from(build_ctx(fns, ti, code), code)
        # mixed-repo stack: same position, fillers from other groups
        pool = [f for gj, f in allfns if gj != gi]
        fillers = rng.sample(pool, len(fns))
        mixed = [dict(f) for f in fillers]
        mixed[ti] = tgt
        gen_m, sc_m = gen_from(build_ctx(mixed, ti, code), code)
        rec = {"repo": g["repo"], "func_name": tgt["func_name"], "n_tok": tgt["n_tok"],
               "target_pos": ti, "solo": sc_s, "same_repo": sc_r, "mixed_repo": sc_m,
               "gen_same": gen_r[:2000], "gen_mixed": gen_m[:2000]}
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        print(f"[{gi+1}/{len(groups)}] {g['repo'][:30]} {tgt['func_name'][:24]:24s} "
              f"solo {sc_s['sim']:5.1f} same {sc_r['sim']:5.1f} mixed {sc_m['sim']:5.1f} "
              f"code {int(sc_s['code'])}/{int(sc_r['code'])}/{int(sc_m['code'])}", flush=True)
out_f.close()
print("done ->", args.out)
print("weights ->", json.dumps(load_meta, sort_keys=True))
