"""Evaluate held-out code questions against latent-vector contexts.

The evaluator joins question and metadata shards positionally, selects held-out
records whose local round-trip grade passed, and draws a seeded comparison set
from trained intents. Model input mirrors training: block-delimited vectors are
followed by input embeddings for ``\nQ: {question}\nA:``. Predictions are graded
with ``dataset.qa_filters.rt_grade``.

``--select-only`` checks the question selection without loading model weights.
The question shards and their source corpus are release-external inputs because
the full training corpus is not distributed with this repository.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.qa_filters import rt_grade, validate_record  # noqa: E402
from dataset.qa_phrase import HELD_OUT_INTENTS  # noqa: E402


def select_records(qa_dir, trained_sample=4000, seed=0):
    """Join numeric question and metadata shards and return evaluation arms."""
    qa_dir = Path(qa_dir)
    files = sorted(p for p in qa_dir.glob("qa_pairs_*.jsonl")
                   if re.fullmatch(r"qa_pairs_\d+\.jsonl", p.name))
    assert files, f"no qa_pairs_<digits>.jsonl in {qa_dir}"
    held, trained_pool = [], []
    stats = {"files": len(files), "rows": 0, "contract_excluded": 0,
             "held_out": 0, "held_intents": {}, "trained_pool": 0}
    for pf in files:
        mf = pf.with_name(pf.name.replace("qa_pairs_", "qa_pairs_meta_", 1))
        assert mf.exists(), f"meta shard missing: {mf.name}"
        plines = pf.read_text().splitlines()
        mlines = mf.read_text().splitlines()
        assert len(plines) == len(mlines), \
            f"{pf.name}: {len(plines)} rows vs meta {len(mlines)} — positional join torn"
        for ln, (lp, lm) in enumerate(zip(plines, mlines)):
            r, m = json.loads(lp), json.loads(lm)
            stats["rows"] += 1
            assert r["held_out"] == (r["intent"] in HELD_OUT_INTENTS), \
                f"{pf.name}:{ln}: held_out flag mismatches intent {r['intent']}"
            if validate_record(r):
                # Contract-invalid records are excluded from every denominator.
                stats["contract_excluded"] += 1
                continue
            r["pair_id"], r["rt_pass"] = m.get("pair_id"), m.get("rt_pass")
            if r["held_out"]:
                assert r["context"]["kind"] == "solo", \
                    f"held-out record {r['pair_id']} has stack context — evaluator " \
                    "only implements the solo (qapre) assembly; extend before running"
                stats["held_out"] += 1
                ic = stats["held_intents"].setdefault(
                    r["intent"], {"n": 0, "rt_true": 0})
                ic["n"] += 1
                ic["rt_true"] += r["rt_pass"] is True
                held.append(r)
            elif m.get("rt_pass") is True and r["context"]["kind"] == "solo":
                trained_pool.append(r)
    stats["trained_pool"] = len(trained_pool)
    heldout_rt_true = [r for r in held if r["rt_pass"] is True]
    heldout_advisory = [r for r in held if r["rt_pass"] is not True]
    rng = random.Random(seed)
    trained_cmp = (rng.sample(trained_pool, min(trained_sample, len(trained_pool)))
                   if trained_sample else [])
    assert not any(r["held_out"] for r in trained_cmp), "held-out leak into comparison arm"
    assert heldout_rt_true, "empty held-out rt_true denominator — wrong artifact dir?"
    return heldout_rt_true, heldout_advisory, trained_cmp, stats


def qa_prompt(rec):
    """the trainer's _qa_prompt_ids text form (options lettered iff MCQ)"""
    q = rec["question"]
    if rec.get("options"):
        q += "".join(f"\n{chr(65 + i)}) {o}" for i, o in enumerate(rec["options"]))
    return f"\nQ: {q}\nA:"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa-data", type=Path, required=True,
                   help="Directory of qa_pairs_*.jsonl and matching metadata shards.")
    p.add_argument("--data", type=Path,
                   help="Unshuffled corpus indexed by each record's row_idx.")
    p.add_argument("--expect-heldout", type=int, default=3983,
                   help="Expected held-out rt_pass=true denominator; 0 disables.")
    p.add_argument("--trained-sample", type=int, default=4000,
                   help="Seeded comparison draw of trained-intent solo records.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--select-only", action="store_true",
                   help="Check selection and counts without loading models.")
    p.add_argument("--no-advisory", action="store_true",
                   help="Skip the separately reported rt_pass=false arm.")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N records in each arm.")
    p.add_argument("--pf", type=int, default=4)
    p.add_argument("--decoder", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--encoder", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--proj-width", type=int, default=None)
    p.add_argument("--proj-depth", type=int, default=2)
    p.add_argument("--pooling", type=str, default="latent",
                   choices=["mean", "latent"],
                   help="Must match the released checkpoint.")
    p.add_argument("--boundary", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bidir-encoder", action="store_true")
    p.add_argument("--projector-weights", type=Path,
                   default=ROOT / "weights" / "projector_ema.safetensors",
                   help="EMA projector weights.")
    p.add_argument("--model-weights", type=Path,
                   default=ROOT / "weights" / "model.safetensors",
                   help="Encoder and decoder weights.")
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--max-new-cap", type=int, default=64,
                   help="Per-record generation cap.")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "qa.jsonl",
                   help="Per-question JSONL receipt.")
    args = p.parse_args()

    heldout, advisory, trained, stats = select_records(
        args.qa_data, trained_sample=args.trained_sample, seed=args.seed)
    print(f"selection: {stats['rows']} rows in {stats['files']} shard pairs | "
          f"held-out {stats['held_out']} {stats['held_intents']} | "
          f"bar denominator (rt_true) {len(heldout)} | advisory {len(advisory)} | "
          f"contract-excluded {stats['contract_excluded']} (135 expected artifact-wide) | "
          f"trained solo rt_true pool {stats['trained_pool']} -> sample {len(trained)}")
    if args.expect_heldout and not args.limit:
        assert len(heldout) == args.expect_heldout, \
            "question artifact drift; inspect the selected shards"
    if args.select_only:
        print("--select-only: done ($0 dry-run, no models loaded)")
        return

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from datasets import load_from_disk
    from compressor import Compressor, ft_load
    from compressor.genutil import batched_generate

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.data is None:
        p.error("--data is required unless --select-only is used")
    corpus = load_from_disk(str(args.data))["train"]
    dec_hidden = AutoConfig.from_pretrained(args.decoder).hidden_size
    comp = Compressor(encoder_name=args.encoder, decoder_hidden=dec_hidden,
                      pooling_factor=args.pf, proj_width=args.proj_width,
                      proj_depth=args.proj_depth, pooling=args.pooling,
                      boundary=args.boundary,
                      bidirectional=args.bidir_encoder).to(device)
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
    embed_table = decoder.get_input_embeddings()

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w", encoding="utf-8")

    @torch.no_grad()
    def run_arm(name, recs):
        if args.limit:
            recs = recs[: args.limit]
        n_ok, n_empty = 0, 0
        by_intent = {}
        for s in range(0, len(recs), args.gen_batch):
            chunk = recs[s: s + args.gen_batch]
            ctxs, budgets = [], []
            for r in chunk:
                assert 0 <= r["row_idx"] < len(corpus), f"row_idx {r['row_idx']} out of range"
                code = corpus[r["row_idx"]]["code"]
                vecs = comp.compress(code, device)[0]  # [<block>]vecs[</block>]
                pids = tok(qa_prompt(r), add_special_tokens=False).input_ids
                pemb = embed_table(torch.tensor(pids, device=device)).to(vecs.dtype)
                ctxs.append(torch.cat([vecs, pemb], dim=0))
                n_ans = len(tok(" " + r["answer"], add_special_tokens=False).input_ids)
                budgets.append(max(16, min(n_ans + 8, args.max_new_cap)))
            outs = batched_generate(decoder, ctxs, budgets, batch_size=args.gen_batch)
            for r, o in zip(chunk, outs):
                pred = tok.decode(o.tolist(), skip_special_tokens=True)
                pred = pred.split("\n", 1)[0].strip()[:200]
                n_empty += not pred
                ok = rt_grade(pred, r)
                n_ok += ok
                d = by_intent.setdefault(r["intent"], [0, 0])
                d[0] += ok
                d[1] += 1
                out_f.write(json.dumps({
                    "arm": name, "pair_id": r["pair_id"], "intent": r["intent"],
                    "source": r["source"], "answer_type": r["answer_type"],
                    "row_idx": r["row_idx"], "rt_pass": r["rt_pass"],
                    "question": r["question"], "answer": r["answer"],
                    "pred": pred, "ok": ok}) + "\n")
            done = min(s + args.gen_batch, len(recs))
            print(f"  [{name}] {done}/{len(recs)}  running {100*n_ok/done:.1f}%",
                  flush=True)
        out_f.flush()
        rate = 100 * n_ok / max(len(recs), 1)
        ibrk = " ".join(f"{i}:{a}/{b}" for i, (a, b) in sorted(by_intent.items()))
        print(f"{name}: {n_ok}/{len(recs)} = {rate:.1f}%  [{ibrk}]"
              + (f"  !! {n_empty} EMPTY predictions !!" if n_empty else ""))
        return n_ok, len(recs), rate

    arms = [("heldout_rt_true", heldout)]
    if trained:
        arms.append(("trained_sample", trained))
    if not args.no_advisory:
        arms.append(("heldout_advisory", advisory))
    res = {name: run_arm(name, recs) for name, recs in arms}
    out_f.write(json.dumps({"_meta": {"argv": sys.argv, "loads": load_meta,
                                      "selection": stats,
                                      "results": {k: v for k, v in res.items()}}}) + "\n")
    out_f.close()

    if args.limit:
        print("\n(SMOKE run — no bar verdict)")
        return
    k, n, rate = res["heldout_rt_true"]
    print(f"\nheld-out QA (rt_pass=true): {rate:.1f}% ({k}/{n})")
    if "trained_sample" in res:
        tr = res["trained_sample"][2]
        delta = tr - rate
        print(f"clause 2 (held-out within 10 pts of trained intents): "
              f"{'PASS' if delta <= 10 else 'FAIL'} — trained {tr:.1f}% vs "
              f"held-out {rate:.1f}% (delta {delta:+.1f} pts; trained records "
              "were seen in training — intent-generalization read)")
    if "heldout_advisory" in res:
        print(f"advisory (rt_pass=false, separate from the primary denominator): "
              f"{res['heldout_advisory'][2]:.1f}% "
              f"({res['heldout_advisory'][0]}/{res['heldout_advisory'][1]})")
    print(f"per-item results -> {out_path}")


if __name__ == "__main__":
    main()
