"""Train the exact-latent reconstruction model.

The decoder regenerates a function's exact tokens from continuous latent
vectors supplied in place of a token prefix:

    [latent vectors][code tokens...]  loss on the code positions only

The primary training diagnostic is the gap between no-context loss and
with-vectors loss: the information carried by the latent vectors.
Evaluation uses functions from held-out repositories.

Run: uv run python train/train_reconstruction.py [--steps N]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from compressor import Compressor

ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--pf", type=int, default=8, help="pooling factor (decoder-slot reduction factor)")
p.add_argument("--lora-r", type=int, default=0,
               help="if >0, LoRA-adapt the DECODER (q/k/v/o, this rank): r9's stage-2 "
                    "co-adaptation. Base weights stay frozen; A/B matrices train "
                    "alongside the projector. 0 = frozen decoder (legacy).")
p.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scale numerator (alpha/r)")
p.add_argument("--lora-lr", type=float, default=1e-4,
               help="separate LR for LoRA params (projector keeps --lr)")
p.add_argument("--init-projector", type=str, default=None,
               help="path to a projector state dict to warm-start from (stage-1 "
                    "alignment artifact, e.g. runs/warmstart/projector_latest.pt)")
p.add_argument("--init-lora", type=str, default=None,
               help="path to a LoRA state dict to warm-start the decoder from "
                    "(e.g. runs/warmstart/lora_ema.pt; requires matching --lora-r)")
p.add_argument("--distractor-frac", type=float, default=0.0,
               help="prob per recon batch of prepending 128 tokens of unrelated code "
                    "as plain text BEFORE the vectors — trains in-flow robustness "
                    "(vectors must work mid-window, not only at position 0)")
p.add_argument("--mask-distractor-pad", action="store_true",
               help="attention-mask the EOS filler tokens that pad a short "
                    "distractor to its fixed 128 slots. Default off: filler "
                    "tokens remain visible, preserving the released training "
                    "distribution unless the recipe explicitly enables masking.")
p.add_argument("--boundary", action="store_true",
               help="learned <block>/</block> embeddings around every vector block "
                    "(the trained form of the load-bearing plaintext labels)")
p.add_argument("--pf-mix", type=str, default=None,
               help="mixed-ratio training, e.g. '4=0.7,8=0.3': each step samples its "
                    "pooling factor (one model, all ratios). Overrides --pf-start.")
p.add_argument("--stack-n-max", type=int, default=6,
               help="stack task samples N uniform in [2, this] functions per context "
                    "(sample count auto-shrinks for large N to bound activation memory)")
p.add_argument("--span-scaffold", type=float, default=0.0,
               help="initial weight of the annealed AST-scaffold loss: KL from "
                    "AST+surprisal-proposed per-span attention mass to the pooler's "
                    "actual allocation. Anneals to zero over --scaffold-steps "
                    "(cosine); no parser is used at inference. 0 = off.")
p.add_argument("--scaffold-steps", type=int, default=5000,
               help="steps over which the scaffold weight cosine-anneals to zero")
p.add_argument("--prose-weight-short", type=float, default=None,
               help="prose-weight override for samples whose docstring is <200 "
                    "characters; a floor helps preserve short verbatim docstrings")
p.add_argument("--stack-token-budget", type=int, default=0,
               help="cap total code tokens per stack context (0 = off). Lets N reach "
                    "stack-n-max without the decoder context outgrowing VRAM: large-N "
                    "stacks draw shorter functions.")
p.add_argument("--eos", action="store_true",
               help="append EOS to recon/stack/qa targets so the model learns to STOP "
                    "(fixes the repeating-tail failure)")
p.add_argument("--proj-width", type=int, default=None, help="projector hidden width")
p.add_argument("--proj-depth", type=int, default=2, help="projector linear layer count")
p.add_argument("--pooling", type=str, default="mean", choices=("mean", "attn", "phase", "latent"),
               help="attn = learned importance weights + slot badges (order-aware); "
                    "phase = RoPE-style within-group rotation before the mean; "
                    "latent = latent-query allocation pooler (learned cross-attention "
                    "over all encoder states, residual on phase-mean, step-0 identity); "
                    "mean = order-blind flat average (legacy)")
p.add_argument("--pooler-lr", type=float, default=None,
               help="separate LR for the latent pooler (default: --lr)")
p.add_argument("--pooler-warmup", type=int, default=0,
               help="train ONLY the latent pooler for this many steps before unfreezing "
                    "projector+LoRA (stabilizes the warm start)")
p.add_argument("--bidir-encoder", action="store_true",
               help="encoder reads bidirectionally. The mask flip changes encoder "
                    "states at step 0 regardless of LoRA B=0; pair with --enc-warmup "
                    "so encoder LoRA repairs the flip before joint training")
p.add_argument("--enc-lora-r", type=int, default=0,
               help="LoRA rank on the encoder's q/k/v/o (0=off); rank 32 recommended")
p.add_argument("--enc-lora-alpha", type=float, default=64.0)
p.add_argument("--enc-lora-lr", type=float, default=1e-4)
p.add_argument("--init-enc-lora", type=str, default=None)
p.add_argument("--enc-warmup", type=int, default=0,
               help="train ONLY the encoder LoRA for this many steps (projector+pooler+"
                    "decoder-LoRA frozen) — task-native LLM2Vec-style repair of the "
                    "bidirectional flip before joint training")
p.add_argument("--surprisal-weight", type=float, default=0.0,
               help=" weight each target token's CE by 1 + k*surprisal_norm using "
                    "the CONTEXTUAL floor-model surprisal column (data/surprisal_train.pt, "
                    "built by the surprisal-data builder). Pays for information, not tokens — the "
                    "'not a democracy' fix. Mutually exclusive with --rare-weight "
                    "(the unigram proxy it replaces). Covers recon/cont/stack targets.")
p.add_argument("--margin-weight", type=float, default=0.0,
               help="decisiveness control: extra CE weight where THIS model's own margin "
                    "(true-token logit minus best rival) is thin or negative — the "
                    "rank-2 commitment failures the static surprisal table cannot see "
                    "(gate sweep: surprisal predicts our misses at <=11%% precision). "
                    "w *= 1 + k*sigmoid(-margin/tau). The difficulty oracle is the "
                    "student itself, recomputed every step; self-annealing (resolved "
                    "ties decay to weight 1). Composes with --surprisal-weight.")
p.add_argument("--margin-tau", type=float, default=2.0,
               help="margin softness (logits); ~2 means ties within a couple logits "
                    "of the rival get most of the extra weight")
p.add_argument("--margin-ramp", type=int, default=500,
               help="steps to ramp margin weight 0->k (early margins are noise)")
p.add_argument("--rank-loss", type=float, default=0.0,
               help="the ranking-hinge design: lambda for the pairwise ranking hinge "
                    "max(0, m* - (z_gold - z_rival)) added to CE on NON-PROSE "
                    "supervised tokens. Optimizes the gold-rival gap DIRECTLY — "
                    "distinct from the dead reweighting family (rare/surprisal/"
                    "margin-weight), which only rescaled CE. Gated by the existing "
                    "prose mask; uniform over non-prose classes (v1.3: one "
                    "mechanism, no class boosts). Requires --chunked-loss.")
p.add_argument("--rank-margin", type=float, default=3.0,
               help="m*: target gold-rival logit gap. 3.0 = the verify21/exp-26 "
                    "flag threshold — train the gap the decode-time flag measures")
p.add_argument("--rank-ramp", type=int, default=1000,
               help="steps to ramp rank-loss lambda 0->target (early margins are "
                    "unconverged-readout noise, same rationale as --margin-ramp)")
p.add_argument("--rank-active-norm", action="store_true",
               help="saturation control: normalize the hinge by the count of "
                    "ACTIVE ties (margin<m*) instead of folding it into the CE "
                    "token-mass denominator. Folded, the term's total gradient "
                    "shrinks as ties get rare — it fades exactly when it is "
                    "finally aimed only at the unsolved residue (the earlier configurations "
                    "endgame). Active-norm keeps the term's total budget "
                    "constant, so per-tie pressure GROWS as the active set "
                    "shrinks. NOTE: per-tie pressure is RANK_K/act_n vs the "
                    "folded RANK_K/den — use a ~10-20x smaller --rank-loss for "
                    "comparable early-training pressure.")
p.add_argument("--rank-recon-only", action="store_true",
               help="saturation control: apply the hinge ONLY on recon batches. "
                    "an earlier configuration ran it on every weighted task (~55%% of hinge budget on "
                    "cont/desc/stack/qa), where a margin-3 demand fights honest "
                    "open-ended uncertainty. In recon the vector determines the "
                    "answer, so a tie there is a retrieval failure by "
                    "construction — the only ties worth paying to close.")
p.add_argument("--hard-mine", type=float, default=0.0,
               help="saturation control, sample level: fraction of each recon "
                    "batch drawn from the hard pool — rows whose recent recon "
                    "batches carried hinge-active tokens, EMA-ranked, class-"
                    "agnostic (the model's own ties pick the rows; no AST). "
                    "Pool is in-memory only (not in checkpoints; refills in "
                    "~1 epoch of recon steps after resume). Adds one GPU sync "
                    "per recon step for the attribution readback. Needs "
                    "--rank-loss and --chunked-loss.")
p.add_argument("--self-prefix-frac", type=float, default=0.0,
               help="decisiveness control, exposure half: fraction of weighted batches "
                    "where the teacher-forced input is corrupted with the model's OWN "
                    "current predictions (extra no-grad forward), so wrong commitments "
                    "are experienced and punished during training instead of being "
                    "erased by the gold prefix every step")
p.add_argument("--self-prefix-p", type=float, default=0.25,
               help="within a corrupted batch: probability each mispredicted target "
                    "position carries the model's token instead of gold (labels stay gold)")
p.add_argument("--self-prefix-maxlen", type=int, default=1600,
               help="skip the exposure forward on batches wider than this (stack "
                    "contexts): its transient + cache blocks OOM'd a 95GB card by "
                    "112MiB twice on validation; exposure matters for recon anyway")
p.add_argument("--task-markers", action="store_true",
               help="prepend a short no-loss text marker to cont ('Continue:') and desc "
                    "('Describe:') contexts. FIXES the task-mode confusion found "
                    "validation: recon and cont contexts are otherwise structurally "
                    "identical, so the decoder GUESSES the task from vector content — "
                    "CoverageConfig (60%% leading prose) pattern-matched 'first half, "
                    "continue it' in every cont-trained run since r7")
p.add_argument("--enc-grad-ckpt", action="store_true",
               help="gradient-checkpoint the encoder (~10x less activation memory for "
                    "~+10%% encoder step time). Needed once enc-LoRA puts the encoder "
                    "in the backward graph: +15-20GB activations at full batch otherwise")
p.add_argument("--mix", type=str, default="recon=1.0",
               help="task mix, e.g. 'recon=0.4,cont=0.15,desc=0.1,plain=0.2,stack=0.15'. "
                    "cont: compress first half of fn, predict second half. desc: compress "
                    "prose-stripped code, predict the docstring. plain: raw-text LM on "
                    "uncompressed code, NO vectors (anti-forgetting rehearsal — only "
                    "meaningful with --lora-r). stack: many functions' vector blocks with "
                    "'### function: name' text headers in ONE context; reproduce the one "
                    "asked for by name (in-flow retrieval). Default = legacy recon-only.")
p.add_argument("--anchor-frac", type=float, default=0.0,
               help="router stage 0 (oracle): this fraction of tokens (highest corpus "
                    "rarity, prose excluded) ride along VERBATIM — each group's pooled "
                    "vector is followed by its anchor tokens' raw decoder embeddings. "
                    "The decoder can copy hard tokens instead of recalling them. 0=off.")
p.add_argument("--decoder", type=str, default="Qwen/Qwen3-1.7B")
p.add_argument("--encoder", type=str, default="Qwen/Qwen3-1.7B",
               help="frozen encoder backbone (must share the decoder's tokenizer)")
p.add_argument("--hf-revision", type=str, default=None,
               help="HF hub revision (commit/tag) pinned for every from_pretrained "
                    "call (config/tokenizer/decoder/encoder/prefetch-worker tokenizer). "
                    "None uses the mutable default branch. Part of the resume args hash.")
p.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (default: auto)")
p.add_argument("--eval-every", type=int, default=250)
p.add_argument("--eval-samples", type=int, default=64)
p.add_argument("--gen-every", type=int, default=1000,
               help="run generation eval (canary exact/similarity) every N steps")
p.add_argument("--gen-canaries", type=int, default=8)
p.add_argument("--gen-batch", type=int, default=8,
               help="canaries generated per batch in GEN eval (left-padded; 1 = bit-exact "
                    "legacy serial). Batch-shape float nondeterminism can flip greedy ties "
                    "(~2/8 canaries, single tokens) — fine for the in-loop TREND gauge, "
                    "which is why 8 is default here but 1 is default in the verdict "
                    "instruments (03/14). Weak-CPU hosts choke on serial loops.")
p.add_argument("--resume", action="store_true", help="resume from state_latest.pt in run dir")
p.add_argument("--rare-weight", type=float, default=0.0,
               help="upweight rare tokens in the training loss by up to this factor (0=off)")
p.add_argument("--mask-p", type=float, default=0.0,
               help="prob of blanking rare tokens from the visible history so spelling "
                    "must come from the vectors (0=off)")
p.add_argument("--pf-start", type=int, default=None,
               help="curriculum: train at this pooling factor first, anneal to --pf")
p.add_argument("--pf-switch", type=int, default=10000,
               help="step at which curriculum switches pf-start -> pf")
p.add_argument("--lr-decay", action="store_true",
               help="cosine-decay LR to 5%% of peak by --steps (default: constant after warmup)")
p.add_argument("--ema", type=float, default=0.0,
               help="EMA decay for projector weights, e.g. 0.999 (0=off). GEN eval runs on the EMA weights.")
p.add_argument("--prose-weight", type=float, default=1.0,
               help="loss weight for docstring/comment tokens (<1 = care less about prose "
                    "wording; frees vector capacity for code). Also excludes prose from "
                    "rarity boosting and closed-book masking. 1.0 = off (legacy).")
p.add_argument("--chunked-loss", action="store_true",
               help="never materialize the full BxTxV logits tensor: grade label positions "
                    "in checkpointed slices (math-identical; removes the memory monster)")
p.add_argument("--loss-chunk", type=int, default=2048,
               help="positions per slice for --chunked-loss")
p.add_argument("--compile", action="store_true",
               help="torch.compile the decoder trunk for the loss path (generate stays eager)")
p.add_argument("--test-equivalence", action="store_true",
               help="verify chunked loss == full loss (values and projector grads), then exit")
p.add_argument("--dec-dtype", type=str, default="bfloat16",
               help="decoder dtype (float32 useful for --test-equivalence: separates "
                    "math bugs from bf16 rounding)")
p.add_argument("--run-name", type=str, default=None)
# --- full fine-tuning ---------------------------------------------------------
p.add_argument("--full-ft", action="store_true",
               help="train ALL encoder+decoder weights (no LoRA — --lora-r/--enc-lora-r "
                    "are auto-zeroed). Param groups: projector @--lr, encoder @--enc-lr, "
                    "decoder @--dec-lr. EMA covers the projector only (full-weight EMA "
                    "would double memory). --resume works via full_state_latest.pt but "
                    "ONLY with byte-identical flags (a changed --steps silently reshapes "
                    "the cosine schedule). --dec-grad-ckpt + --enc-grad-ckpt are "
                    "memory-mandatory at real batch sizes.")
p.add_argument("--enc-lr", type=float, default=1e-5,
               help="full-ft encoder learning rate (C3 used 1e-5 for both models)")
p.add_argument("--dec-lr", type=float, default=1e-5,
               help="full-ft decoder learning rate")
p.add_argument("--ft-dtype", type=str, default="float32", choices=["float32", "bfloat16"],
               help="full-ft weight dtype. float32 = master weights + autocast-bf16 compute "
                    "(correct mixed precision, ~57GB pre-activation for 1.7B+4B); bfloat16 = "
                    "pure-bf16 torchtune-style (~35GB, per-step rounding risk — validation tier)")
p.add_argument("--adam-8bit", action="store_true",
               help="bitsandbytes AdamW8bit (moment buffers 8-bit, embeddings+lm_head kept "
                    "32-bit state). CUDA only. The lever that fits full-ft on one 94GB card.")
p.add_argument("--dec-grad-ckpt", action="store_true",
               help="gradient checkpointing on the decoder (use_reentrant=False); mandatory "
                    "for full-ft at real batch sizes")
p.add_argument("--grad-accum", type=int, default=1,
               help="micro-batches per optimizer step (effective batch = batch x this). "
                    "--steps counts OPTIMIZER steps; scheduler/EMA/telemetry follow them.")
p.add_argument("--grad-clip", type=float, default=0.0,
               help="global grad-norm clip; 0 = off (legacy). Recommend 1.0 for full-ft.")
p.add_argument("--full-save-every", type=int, default=1000,
               help="full-ft resume-state cadence in optimizer steps. With fp32 Adam the "
                    "save is ~68GB (fp32 weights 23 + fp32 m/v 45) and tmp+old coexist "
                    "during the atomic rename — provision >=250GB free disk. Each save "
                    "costs ~1-2 min; at 1000 that's a crash-loss ceiling of ~1000 steps. "
                    "NOTE: the save fires inside the eval block — --eval-every must "
                    "divide this (asserted at startup).")
# --- distributed training: DDP + ZeRO-1 --------------------------------------
p.add_argument("--distributed", action="store_true",
               help="DDP + ZeroRedundancyOptimizer across torchrun ranks. With this flag "
                    "ABSENT the single-process path is byte-identical to the legacy "
                    "trainer. Launch: uv run torchrun --standalone --nproc_per_node=N "
                    "train/train_reconstruction.py --distributed ...")
p.add_argument("--eff-batch", type=int, default=None,
               help="assert world*batch*grad-accum equals this (launch-typo guard)")
# --- FSDP2 distributed training ----------------------------------------------
p.add_argument("--fsdp", action="store_true",
               help="FSDP2 (fully_shard, FULL_SHARD) instead of DDP+ZeRO-1 under "
                    "--distributed: params/grads/optimizer state sharded per rank, "
                    "and grads reduce-scatter every micro-batch so accumulation "
                    "stays sharded (~1.7GB versus DDP's 13.4GB replicated fp32 "
                    "gradients). The DDP+ZeRO-1 path remains available as a fallback.")
p.add_argument("--fsdp-reshard", type=str, default="full", choices=("full", "grad_op"),
               help="full = FULL_SHARD, blocks reshard params after forward (F3 "
                    "default; root group never reshards after forward). grad_op = "
                    "SHARD_GRAD_OP diagnostic fallback: params stay unsharded "
                    "fwd->bwd (+10.8GiB/rank; isolates resharding-path bugs — it "
                    "buys back NO correctness, F3).")
p.add_argument("--fsdp-param-dtype", type=str, default="float32",
               choices=("float32", "bfloat16"),
               help="bfloat16 = MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype="
                    "fp32) on the 54 BLOCK groups only: bf16 all-gather/compute "
                    "for the transformer blocks (2.7B of 3.37B params), fp32 "
                    "sharded masters AND fp32 grad reduction (F5 lever: "
                    "~-5.7GiB/rank, ~-27%% NVLink traffic, module-level numerics "
                    "deviation from the autocast lineage). The ROOT group "
                    "(embeddings/lm_head/norms/projector) stays fp32 — eval "
                    "forwards run outside autocast and the projector/pooler is "
                    "an fp32-by-design island; bf16 root params crash strict-"
                    "dtype kernels there (validation). float32 = no "
                    "MixedPrecisionPolicy; autocast-only, numerics identical to "
                    "the validated reference path (default).")
p.add_argument("--fused-adam", action="store_true",
               help="fused AdamW kernel under --fsdp (safe on torch 2.13: the "
                    "2.9-2.12 per-step DTensorSpec host-RAM leak #188174 is fixed "
                    "there). Default off; enable only after CUDA validation (F8).")
p.add_argument("--fsdp-no-count-sync", action="store_true",
               help="TEST ONLY (negative-control validation): disable the F4 "
                    "forward-count sync so a rank-divergent stack draw HANGS. "
                    "Proves the sync is load-bearing. Never set in production.")
p.add_argument("--telemetry-batched", action="store_true",
               help="accumulate MARGIN_STATS/TB_STATS in on-device tensors and "
                    "drain to Python only at the existing logging boundaries "
                    "(%%10 steps.jsonl write / %%50 step line) — removes up to "
                    "~25 per-micro host syncs (8xH100 profile: the per-chunk "
                    "float()/int() reads dominated the fixed step tail). Same "
                    "keys, same cadence; values may differ only by fp32 "
                    "accumulation order. Default off = legacy per-chunk path.")
p.add_argument("--fsdp-accum-hold", action="store_true",
               help="keep params GATHERED across the grad-accum window: micro 0 "
                    "disables reshard-after-forward AND reshard-after-backward "
                    "on the 54 block groups + root, so each group all-gathers "
                    "ONCE per window instead of 2x per micro (fwd + bwd; the "
                    "grad-ckpt recompute re-gather also disappears). After "
                    "opt.step() every group is explicitly reshard()ed and "
                    "flags re-armed (eval paths see normal behavior). Grads "
                    "are untouched: reduce-scatter fires EVERY micro and "
                    "accumulates SHARDED fp32 — the leg-3 OOM protection (F2/"
                    "R8). Memory ~= --fsdp-reshard grad_op (+~10.8GiB/rank, "
                    "held through the opt step).")
p.add_argument("--fsdp-hold-no-reshard", action="store_true",
               help="TEST ONLY (negative control): skip the post-step "
                    "reshard under --fsdp-accum-hold so the stale-param "
                    "tripwire must abort. Never set in production.")
p.add_argument("--fsdp-test-bf16-reduce", action="store_true",
               help="TEST ONLY (negative control): build the bf16 "
                    "MixedPrecisionPolicy WITHOUT explicit reduce_dtype=fp32 — "
                    "the F5/#186998 trap (gradients reduce in bf16, final "
                    "dtype still fp32, invisible to the grad-dtype assert). "
                    "The first-backward reduce-dtype introspection must abort. "
                    "Never set in production.")
# --- FSDP speed levers (all default-off) -------------------------------------
p.add_argument("--build-prefetch", action="store_true",
               help="lever D: overlap the CPU half of the NEXT micro-batch's "
                    "build (row draw + dataset fetch + tokenization + Python "
                    "prep) with the current micro's GPU work, on ONE worker "
                    "thread. The GPU half (H2D, comp() forward, assembly) stays "
                    "on the main thread in original order, as do the F4 "
                    "_sync_any/_sync_fwd_count collectives. Requires "
                    "--distributed (every draw is a pure fn of (step,micro,"
                    "rank) via _dstream, so lookahead cannot desync ranks) and "
                    "--hard-mine 0 (its row swap reads prior-loss state — "
                    "incompatible with lookahead). The worker consumes NO torch "
                    "RNG and uses its OWN tokenizer instance; batches are "
                    "bit-identical to the non-prefetch path (a cache miss just "
                    "tokenizes on the main thread — speed-only layer). "
                    "Telemetry: steps.jsonl 'pfetch_r0' (tok_hit==0 = dead cache; "
                    "rank-0-only by name).")
p.add_argument("--ckpt-token-threshold", type=int, default=0,
               help="lever E: per model forward, if the padded token count "
                    "(batch x padded seq len) entering that model is below N, "
                    "disable gradient checkpointing for that forward; else keep "
                    "it. 0 = always-ckpt (current behavior). Applies to "
                    "whichever of encoder/decoder has grad-ckpt enabled; the "
                    "gate is a per-forward pre-hook flipping the HF per-layer "
                    "'gradient_checkpointing' bool (no rebuild). Recompute-vs-"
                    "store is exact (dropout 0) and the FSDP2 per-block "
                    "collective schedule is unchanged either way (rehearsal "
                    "an earlier configuration, rank-asymmetric gating included). Telemetry: "
                    "steps.jsonl 'ckpt' on/off counts per model.")
p.add_argument("--pad-to-max", action="store_true",
               help="lever F: pad encoder batches to max-in-batch rounded UP "
                    "to a multiple of 16 instead of the shape bucket, and round "
                    "assembled decoder lengths up to a multiple of 16 (they "
                    "were already max-in-batch-tight). Eager only — REFUSED "
                    "with --compile (buckets are the compile shape-family "
                    "strategy). Measured on the 8xH100 measurements: buckets waste "
                    "~29%% of tokens / ~50%% of attention FLOPs as padding. "
                    "Realized batches are mathematically identical (padding is "
                    "masked everywhere: attention mask 0, labels -100), but "
                    "--mask-p's torch.rand(enc_ids.shape) consumes RNG shape-"
                    "dependently, so the realized mask stream differs from "
                    "bucketed runs — statistically neutral, documented in the fixed recipe. "
                    "Telemetry: steps.jsonl 'pad' width stats (nm16 must be 0).")
p.add_argument("--eval-autocast", action="store_true",
               help="lever G: run evaluate()/gen_eval()/rank_probe() forwards "
                    "under the same bf16 autocast ctx training uses (they "
                    "currently run true fp32 outside FT_AC — no TF32 on the "
                    "node, ~$90-100 of a 20k run). Identical on all ranks "
                    "(replicated eval). Metric shift class: bf16 rounding on "
                    "eval numbers only; training math untouched.")
# --- pregenerated QA arms and long-row support -------------------------------
p.add_argument("--data", type=str, default=None,
               help="dataset directory for load_from_disk (train/validation splits). "
                    "Default: data/csn_python; the final recipe uses "
                    "data/corpus_v2/corpus.")
p.add_argument("--qa-data", type=str, default=None,
               help="dir of qa_pairs_*.jsonl (pregenerated question/answer records "
                    "indexing the UNSHUFFLED train split by row_idx). Enables the "
                    "qapre/retr/span mix arms. held_out records are eval-only and "
                    "never enter the sampling pools (leak tripwire logged).")
p.add_argument("--qapre-conc-frac", type=float, default=0.0,
               help="fraction of qapre draws forced to the conceptual pool "
                    "(C1/C3 records). 0 leaves them in the general qapre pool. "
                    "The final task mix uses qapre=0.15 with this set to 0.2, "
                    "yielding 12% general QA and 3% conceptual QA in expectation.")
p.add_argument("--long-buckets", action="store_true",
               help="extend shape buckets with 1024/2048/4096. Required for the "
                    "final corpus, whose 512-4096-token long tail would otherwise "
                    "be rejected by the legacy 512-token bucket cap.")
p.add_argument("--token-budget", type=int, default=0,
               help="cap sum(n_tokens) per drawn micro-batch: the row window (and qa "
                    "record draws) shrink until the sum fits (min 1 row), so 4096-token "
                    "rows train in tiny batches instead of OOMing. 0 = off (legacy). "
                    "NOTE: makes the realized effective batch VARIABLE (<= world x "
                    "batch x accum); per-step realized rows/tokens go to steps.jsonl.")
p.add_argument("--dist-fixture", action="store_true",
               help="test only: key content draws by global micro index instead of "
                    "rank so ws=1 x accum=N and ws=N x accum=1 see identical row sets "
                    "(the R2 grad-parity fixture)")
p.add_argument("--force-qa-variant", type=float, default=None,
               help="test only (distributed): pin the shared qa sub-variant coin "
                    "(e.g. 0.1 = raw-text) to force the D6 remap path")
p.add_argument("--dump-grads", type=str, default=None,
               help="test only: after the first stepped backward(+clip), rank 0 "
                    "saves {param name: grad} here and all ranks exit cleanly")
p.add_argument("--dump-grads-raw", type=str, default=None,
               help="test only: like --dump-grads but BEFORE clip_grad_norm_ "
                    "— the UNCLIPPED gradients the production step actually "
                    "applies when --grad-clip is 0 (validation: "
                    "post-clip dumps normalize away uniform scaling errors). "
                    "May be combined with --dump-grads (both dumps, then exit).")
p.add_argument("--halt-after-step", type=int, default=None,
               help="test only: clean stop after this optimizer step (simulated "
                    "interruption for resume tests; excluded from the resume args-hash)")
p.add_argument("--torch-profile", type=str, default=None, metavar="A:B",
               help="observational: rank-0 kineto trace (CPU+CUDA) over optimizer "
                    "steps A..B inclusive (window capped at 10 steps — traces grow "
                    "multi-GB past that). Writes torch_trace_A_B.json.gz + a "
                    "key_averages table to the run dir/stdout. Profiled steps' "
                    "t_step carries profiler overhead — never feed them to the §3 "
                    "estimator. Excluded from the resume args-hash (interruption "
                    "control, like --halt-after-step).")
args = p.parse_args()

if args.full_ft:
    if args.lora_r > 0 or args.enc_lora_r > 0:
        print(f"--full-ft: auto-zeroing LoRA flags (were r={args.lora_r}/enc r={args.enc_lora_r})")
        args.lora_r = 0
        args.enc_lora_r = 0
    args.dec_dtype = args.ft_dtype
    assert args.full_save_every % args.eval_every == 0, \
        f"--full-save-every {args.full_save_every} must be a multiple of --eval-every " \
        f"{args.eval_every} (the save fires inside the eval block)"

# --- distributed init (D11): default-off; legacy path byte-identical without it
import hashlib
import os

DIST = args.distributed
FSDP_ON = DIST and args.fsdp
RANK, WORLD, LOCAL_RANK = 0, 1, 0
assert not (args.fsdp and not DIST), "--fsdp modifies the --distributed path; add --distributed"
assert not (args.fsdp_accum_hold and not FSDP_ON), \
    "--fsdp-accum-hold modifies the --fsdp path; add --distributed --fsdp"
if not DIST and int(os.environ.get("WORLD_SIZE", "1")) > 1:
    sys.exit("FATAL: torchrun world > 1 without --distributed — N independent copies "
             "would fight over one run dir. Add --distributed or drop torchrun.")
# --- speed-lever purity guards (round 2, checked before any dist init) --------
if args.build_prefetch:
    if not DIST:
        sys.exit("FATAL: --build-prefetch requires --distributed — only there is "
                 "every control/content draw a pure function of (step, micro, "
                 "rank) via _dstream; the legacy path's long-lived RNG streams "
                 "cannot be prefetched without reordering consumption.")
    if args.hard_mine > 0:
        sys.exit("FATAL: --build-prefetch is incompatible with --hard-mine — the "
                 "hard-mine row swap reads prior-loss state (HARD_EMA) at build "
                 "time, which a lookahead build cannot see. Set --hard-mine 0 "
                 "(the final training run recipe already does) or drop --build-prefetch.")
if args.pad_to_max and args.compile:
    sys.exit("FATAL: --pad-to-max is refused with --compile — the shape buckets "
             "ARE the compile shape-family strategy; max-in-batch padding would "
             "recompile per batch. Eager only.")
if DIST:
    assert not args.adam_8bit, \
        "--distributed is the fp32-purity path; --adam-8bit is its single-card fallback — never both"
    if args.full_ft:
        assert args.ft_dtype == "float32", \
            "--distributed --full-ft: ZeRO shards need one dense dtype; fp32 masters are the point"
    assert args.device != "mps", "--distributed: MPS has no distributed backend (CUDA or CPU/gloo)"
    if args.fsdp:
        # FSDP purity guards (FSDP DESIGN F12)
        assert not args.compile, \
            "--fsdp: torch.compile was disqualified (leg-3 prof_c1) and FSDP2 + " \
            "non-tensor forward args + our recompile surface is not a $32/hr battle"
        assert args.self_prefix_frac == 0, \
            "--fsdp: --self-prefix-frac gates an EXTRA dec_trunk forward on an " \
            "unseeded rank-local coin — a guaranteed collective-schedule hang " \
            "(F4). Make its coin a shared draw before re-enabling."
        assert not args.fsdp_hold_no_reshard or args.fsdp_accum_hold, \
            "--fsdp-hold-no-reshard is the --fsdp-accum-hold negative control"
        assert not args.fsdp_test_bf16_reduce or args.fsdp_param_dtype == "bfloat16", \
            "--fsdp-test-bf16-reduce sabotages the bf16 MixedPrecisionPolicy; " \
            "it needs --fsdp-param-dtype bfloat16"
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        sys.exit("FATAL: --distributed requires the torchrun env (RANK/WORLD_SIZE/LOCAL_RANK). "
                 "Launch: uv run torchrun --standalone --nproc_per_node=N "
                 "train/train_reconstruction.py --distributed ...")
    from datetime import timedelta

    import torch.distributed as dist
    RANK = int(os.environ["RANK"])
    WORLD = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    # 45 min: the ~68GB rank-0 save takes 1-2 min and pathological evals more;
    # the default 10 min would turn a slow eval into a poisoned NCCL group
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
        # device_id binds the NCCL group; gloo/CPU must NOT pass it (macOS ValueError).
        # FSDP adds a CPU backend alongside NCCL: dcp.async_save hard-requires one
        # ("A CPU backend must be enabled for async save", state_dict_saver.py) —
        # the DDP fallback keeps the pure-NCCL init it was shaken down on.
        _backend = "cpu:gloo,cuda:nccl" if args.fsdp else "nccl"
        dist.init_process_group(_backend, device_id=torch.device(f"cuda:{LOCAL_RANK}"),
                                timeout=timedelta(minutes=45))
        args.device = f"cuda:{LOCAL_RANK}"
    else:
        dist.init_process_group("gloo", timeout=timedelta(minutes=45))
        args.device = "cpu"
IS_MAIN = RANK == 0
if DIST and not IS_MAIN:
    def print(*a, **k):  # noqa: A001 — rank 0 owns stdout; rank errors surface as exceptions
        pass
if args.eff_batch is not None:
    assert WORLD * args.batch * args.grad_accum == args.eff_batch, \
        f"eff batch {WORLD}x{args.batch}x{args.grad_accum} != --eff-batch {args.eff_batch}"

# --- FSDP telemetry + forward-count sync (FSDP DESIGN F4/F10, log-everything) --
# FSDP2 derives its backward-prefetch collective schedule from how many times
# each wrapped module ran forward, so a rank-divergent comp() invocation count
# HANGS (measured, both reshard strategies — DESIGN §0). Every rank-local draw
# that changes a wrapped module's call count must be reconciled first.
# count_div_events is both the leading indicator and the no-op tripwire
# (0 forever = symmetric data OR a dead mechanism; validation's negative
# control distinguishes them); count_div_max prices the min-truncation.
FSDP_STATS = {"count_div_events": 0, "count_div_max": 0, "rebuild_sync": 0,
              "qa_fallback_sync": 0, "not_run_forward_warns": 0, "bwd_ms": 0.0,
              # --fsdp-accum-hold leading indicator: windows held (== optimizer
              # steps while the flag is live; 0 with the flag on = dead
              # mechanism). Its no-op tripwire is the post-step sharded assert.
              "hold_windows": 0}
# The F4 sync collectives are called ONLY on the arms that need them — safe
# because the task draw is SHARED (rank-identical per micro) in normal
# distributed mode. --dist-fixture deliberately makes tasks RANK-VARYING
# (each rank replays one global micro of the reference run), which (a) pairs
# mismatched all-reduces across ranks -> gloo hang (observed: R2-mixed ws=4
# stuck 16 min at 2 steps), and (b) would BREAK grad parity anyway — any
# cross-rank reconciliation changes the data vs the ws=1 reference. So under
# the fixture every F4 sync is disabled; the fixture configs must therefore
# keep comp()-call counts rank-uniform by construction (1 call per vector arm,
# no 'plain', no forced qa-raw — the harness R2/R6b mixes satisfy this).
FSDP_SYNC = FSDP_ON and not args.dist_fixture


def _sync_fwd_count(n):
    """all-reduce a rank-local count to its cross-rank MIN (rank-identical).
    Collective: every rank on the arm must call it exactly once per use site."""
    if not FSDP_SYNC or args.fsdp_no_count_sync:
        return n  # no-count-sync: TEST ONLY — R7's must-hang negative control
    t = torch.tensor([n, -n], dtype=torch.int64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)  # -> (min_n, -max_n)
    lo, hi = int(t[0]), int(-t[1])
    if hi != lo:
        FSDP_STATS["count_div_events"] += 1
        FSDP_STATS["count_div_max"] = max(FSDP_STATS["count_div_max"], hi - lo)
    return lo


def _sync_any(flag):
    """all-reduce a rank-local bool to 'any rank' (shared decision). Collective."""
    t = torch.tensor([int(bool(flag))], dtype=torch.int64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(int(t))


if FSDP_ON:
    # tripwire for a silently skipped wrap unit (F10). Two layers:
    # (1) log filter for torch<=2.12's warning_once ("... did not run forward
    #     before backward ...", _fsdp_state.py). Torch 2.13 removed that
    #     warning and silently completes skipped units, so string matching is
    #     no longer sufficient.
    # (2) the load-bearing replacement: an introspective forward-coverage
    #     counter (hooks installed after the wrap, drained at the %10
    #     telemetry boundary): any wrapped BLOCK that ran zero grad-enabled
    #     training forwards in a logging window increments the same nrf
    #     counter. The all-'plain' negative control, where encoder blocks
    #     never run, must drive nrf above zero.
    import logging as _logging

    class _NRFCounter(_logging.Filter):
        def filter(self, record):
            if "did not run forward before backward" in record.getMessage():
                FSDP_STATS["not_run_forward_warns"] += 1
            return True

    _logging.getLogger("torch.distributed.fsdp.fully_shard").addFilter(_NRFCounter())
_BLK_FWD_COV = []  # per-block grad-enabled training-forward counts (F10 layer 2)


def _nrf_cov_drain():
    """count blocks with ZERO training forwards since last drain -> nrf.
    Rank-local like the rest of FSDP_STATS; called at the %10 boundary
    (window = 10 optimizer steps x accum micros — a legitimately all-'plain'
    window at mix plain=0.10 has probability ~0.1^80, never in practice)."""
    if not _BLK_FWD_COV:
        return
    FSDP_STATS["not_run_forward_warns"] += sum(1 for c in _BLK_FWD_COV if c == 0)
    for i in range(len(_BLK_FWD_COV)):
        _BLK_FWD_COV[i] = 0

# resume contract (D8): byte-identical recipe, enforced — not help-text. Excluded:
# interruption controls (halt/dump/resume/device) plus run_name (a path, and its
# default fills in AFTER this hash) and eff_batch (an assertion, not a recipe) —
# they legitimately differ across the interrupt->resume pair. WORLD is INCLUDED:
# resuming at a different world size silently reshapes the effective batch and
# every rank-keyed data stream.
ARGS_HASH = hashlib.md5(json.dumps(
    {"world_size": WORLD,
     **{k: str(v) for k, v in sorted(vars(args).items())
        if k not in ("resume", "device", "halt_after_step", "dump_grads",
                     "dump_grads_raw", "torch_profile", "run_name", "eff_batch")}},
    sort_keys=True).encode()).hexdigest()

DEC_NAME = args.decoder
if args.run_name is None:
    dec_tag = DEC_NAME.split("/")[-1].replace("Qwen3-", "q3-").lower()
    args.run_name = f"recon_pf{args.pf}_{dec_tag}_projonly"

device = args.device or (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
run_dir = ROOT / "runs" / args.run_name
metrics_f = None
if IS_MAIN:  # D9: run dir, metrics fd, config append, git call are rank-0-only
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(run_dir / "metrics.jsonl", "a")

# --- durable run config (the earlier configuration launch command was never recorded — never again)
import subprocess

if IS_MAIN:
    try:
        _git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        _git = ""
    if not _git:
        # staged node workspaces have no .git — the code bundle ships a REV
        # file (the release bundle) so telemetry never records an
        # empty revision again (validation; single-rank validation recorded "")
        _rev_f = ROOT / "REV"
        _git = _rev_f.read_text().strip() if _rev_f.exists() else "unknown"
    with open(run_dir / "config.json", "a") as _cf:
        json.dump({"argv": sys.argv, "args": {k: str(v) for k, v in vars(args).items()},
                   "git": _git, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, _cf)
        _cf.write("\n")
    print(f"run config -> {run_dir/'config.json'} (git {_git})")

# --- models ------------------------------------------------------------------
from transformers import AutoConfig

# --hf-revision: one pin for every hub resolution (— without this the
# fixed recipe HF-revision line was a promise no code enforced). None = mutable
# default branch, byte-identical legacy behavior.
_HF_REV = {"revision": args.hf_revision} if args.hf_revision else {}
dec_hidden = AutoConfig.from_pretrained(DEC_NAME, **_HF_REV).hidden_size
comp = Compressor(encoder_name=args.encoder, decoder_hidden=dec_hidden, pooling_factor=args.pf,
                  proj_width=args.proj_width, proj_depth=args.proj_depth,
                  pooling=args.pooling, boundary=args.boundary,
                  bidirectional=args.bidir_encoder,
                  dtype=getattr(torch, args.ft_dtype) if args.full_ft else torch.bfloat16,
                  hf_revision=args.hf_revision).to(device)
tok = AutoTokenizer.from_pretrained(DEC_NAME, **_HF_REV)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
decoder = AutoModelForCausalLM.from_pretrained(
    DEC_NAME, dtype=getattr(torch, args.dec_dtype), **_HF_REV).to(device)
decoder.requires_grad_(False)
decoder.eval()
embed_table = decoder.get_input_embeddings()
PF = comp.pooling_factor

ft_enc_ps, ft_dec_ps = [], []
if args.full_ft:
    comp.encoder.requires_grad_(True)
    decoder.requires_grad_(True)
    ft_enc_ps = [q for q in comp.encoder.parameters() if q.requires_grad]
    ft_dec_ps = [q for q in decoder.parameters() if q.requires_grad]
    if args.dec_grad_ckpt:
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.enc_grad_ckpt:  # normally wired inside the enc-LoRA block, absent here
        comp.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    n_e = sum(q.numel() for q in ft_enc_ps)
    n_d = sum(q.numel() for q in ft_dec_ps)
    print(f"FULL-FT: encoder {n_e/1e9:.2f}B @ lr {args.enc_lr} + decoder {n_d/1e9:.2f}B "
          f"@ lr {args.dec_lr}, dtype {args.ft_dtype}, dec-grad-ckpt {args.dec_grad_ckpt}, "
          f"eff batch {args.batch * args.grad_accum}")

import contextlib

if args.full_ft and args.ft_dtype == "float32":
    # fp32 master weights + bf16 compute: the correct single-card mixed precision
    def FT_AC():
        return torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)
else:
    def FT_AC():
        return contextlib.nullcontext()

if args.eval_autocast:
    # lever G: eval/gen/probe forwards under the SAME bf16 autocast training
    # uses. Replicated eval => identical decision + ops on every rank.
    def EVAL_AC():
        return torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)
else:
    def EVAL_AC():
        return contextlib.nullcontext()

lora_ps = []
if args.lora_r > 0:
    from compressor.lora import inject_lora, load_lora_state_dict, lora_parameters, lora_state_dict

    sites = inject_lora(decoder.model, args.lora_r, args.lora_alpha)
    lora_ps = lora_parameters(decoder)
    n_lora = sum(p.numel() for p in lora_ps)
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha}: {len(sites)} linears, "
          f"{n_lora/1e6:.1f}M trainable params (B=0 init: step-0 decoder == frozen base)")
    if args.init_lora:
        load_lora_state_dict(decoder, torch.load(args.init_lora, map_location=device))
        print(f"LoRA warm-started from {args.init_lora}")

enc_lora_ps = []
if args.enc_lora_r > 0:
    from compressor.lora import inject_lora, load_lora_state_dict, lora_parameters, lora_state_dict

    enc_sites = inject_lora(comp.encoder, args.enc_lora_r, args.enc_lora_alpha)
    enc_lora_ps = lora_parameters(comp.encoder)
    print(f"encoder LoRA r={args.enc_lora_r} alpha={args.enc_lora_alpha}: "
          f"{len(enc_sites)} linears, {sum(p.numel() for p in enc_lora_ps)/1e6:.1f}M params "
          f"(B=0: step-0 encoder output == frozen base"
          + (", but BIDIRECTIONAL flip moves states regardless" if args.bidir_encoder else "")
          + ")")
    if args.init_enc_lora:
        load_lora_state_dict(comp.encoder, torch.load(args.init_enc_lora, map_location=device))
        print(f"encoder LoRA warm-started from {args.init_enc_lora}")
    if args.enc_grad_ckpt:
        # use_reentrant=False: grads reach the LoRA params inside checkpointed
        # segments even though every checkpoint-segment INPUT is frozen/no-grad
        comp.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("encoder gradient checkpointing ON")

if args.init_projector:
    # strict=False: a boundary-enabled projector has 2 params the warm-start
    # checkpoint predates — they keep their fresh init.
    miss = comp.projector.load_state_dict(
        torch.load(args.init_projector, map_location=device), strict=False)
    print(f"projector warm-started from {args.init_projector}"
          + (f" (fresh: {sorted(miss.missing_keys)})" if miss.missing_keys else ""))

# Loss path runs trunk + lm_head separately (so the trunk can be compiled and the
# loss chunked); generate() keeps using the eager `decoder` — compiling it would
# recompile per KV-cache length and eat the savings.
dec_trunk = decoder.model
lm_head = decoder.get_output_embeddings()
if args.compile:
    # 4 buckets x 2 pf x train/eval shapes; --long-buckets nearly doubles the families
    torch._dynamo.config.cache_size_limit = 128 if args.long_buckets else 64
    dec_trunk = torch.compile(dec_trunk)

# --- lever E: --ckpt-token-threshold — per-forward grad-ckpt gate --------------
# transformers 5.x gates checkpointing PER LAYER: GradientCheckpointingLayer.
# __call__ checks `self.gradient_checkpointing and self.training` and wraps
# super().__call__ in the stored _gradient_checkpointing_func. The cheapest
# per-micro toggle is therefore flipping that bool on the (few dozen) layer
# modules — no rebuild, no re-partial; _gradient_checkpointing_func stays set.
# The gate is a forward PRE-hook on each model trunk: small padded batches
# (B x padded_T < N) skip recompute (peak mem measured 22.2/80 GB with full
# ckpt — typical micros don't need it), big ones keep it. Each model gates on
# ITS OWN padded input size (the encoder runs during build, before the decoder
# length exists; each model's activation memory is driven by its own input).
# FSDP2 invariance: ckpt only moves WHERE the one backward all-gather per block
# fires (recompute pre-forward vs pre-backward hook) — same collectives, same
# per-block order, hold on or off; rank-asymmetric gating is safe (an earlier configuration).
CKPT_N = args.ckpt_token_threshold
CKPT_STATS = {"dec_on": 0, "dec_off": 0, "enc_on": 0, "enc_off": 0}
CKPT_LAST = {"dec": None, "enc": None}  # last training-mode decision (probe)
# (validation): the gate must price the micro's TOTAL per-model token
# footprint, not one forward call — build_stack_ctx splits a 40-row stack into
# ceil(N/16) encoder forwards whose activations COEXIST in one autograd graph,
# so per-call B*T would let each 16-row chunk skip recompute while their
# combined stored footprint is ~N_rows x padded_T. A chunking call site
# ANNOUNCES its whole-micro footprint here before the chunk loop; the gate
# uses the announced total when present, else the call's own B*T (identical
# for every single-forward builder). Announce is set/cleared synchronously
# around the loop on the main thread — no bleed into other micros.
CKPT_ANN = {"dec": 0, "enc": 0}
CKPT_LAST_TOT = {"dec": 0, "enc": 0}  # gate input of the last training decision
CKPT_LAST_CALL = {"dec": 0, "enc": 0}  # that call's OWN B*T (probe: call<N<=tot
# with gate ON is the announced-total signature — per-call logic would gate off)
_CKPT_PROBE = os.environ.get("CKPT_TOGGLE_PROBE") == "1"  # validation only
_CKPT_PROBE_CNT = [0, 0]  # [dec layer0, enc layer0] forward invocations
if CKPT_N > 0:
    def _install_ckpt_gate(root, tag):
        mods = [m for m in root.modules() if getattr(m, "gradient_checkpointing", False)]
        if not mods:
            return False  # grad-ckpt never enabled on this model — nothing to gate

        def _gate(module, hargs, hkwargs):
            x = hkwargs.get("inputs_embeds")
            if x is None:
                x = hkwargs.get("input_ids")
            if x is None and hargs:
                x = hargs[0]
            if x is None or x.dim() < 2:
                return  # unknown call shape: leave ckpt at its enabled default
            tot = CKPT_ANN[tag] or x.shape[0] * x.shape[1]  # micro total ()
            on = not (module.training and tot < CKPT_N)
            for m_ in mods:
                m_.gradient_checkpointing = on
            if module.training:
                CKPT_STATS[tag + ("_on" if on else "_off")] += 1
                CKPT_LAST[tag] = on
                CKPT_LAST_TOT[tag] = tot  # probe: the gate's actual input (an earlier configuration)
                CKPT_LAST_CALL[tag] = x.shape[0] * x.shape[1]
        root.register_forward_pre_hook(_gate, with_kwargs=True)
        return True

    _g_dec = _install_ckpt_gate(decoder.model, "dec")
    _g_enc = _install_ckpt_gate(comp.encoder, "enc")
    assert _g_dec or _g_enc, \
        "--ckpt-token-threshold set but gradient checkpointing is not enabled " \
        "on any model (need --dec-grad-ckpt and/or --enc-grad-ckpt) — the gate " \
        "would be a silent no-op"
    print(f"ckpt-token-threshold {CKPT_N}: gate installed on "
          f"{'dec ' if _g_dec else ''}{'enc' if _g_enc else ''} "
          f"(padded B*T < {CKPT_N} skips recompute)")
if _CKPT_PROBE:
    # an earlier configuration probe: count layer-0 forward INVOCATIONS via module pre-hooks —
    # checkpoint recompute re-enters the layer's __call__, so a ckpt'd micro
    # shows 2x the forwards of a gated-off one. Test-only (env), zero default
    # impact.
    decoder.model.layers[0].register_forward_pre_hook(
        lambda m, a: _CKPT_PROBE_CNT.__setitem__(0, _CKPT_PROBE_CNT[0] + 1))
    comp.encoder.layers[0].register_forward_pre_hook(
        lambda m, a: _CKPT_PROBE_CNT.__setitem__(1, _CKPT_PROBE_CNT[1] + 1))

# --- data ---------------------------------------------------------------------
if DIST and not IS_MAIN:
    dist.barrier()  # rank 0 materializes the shuffle caches first (torn-write race)
data = load_from_disk(args.data or str(ROOT / "data" / "csn_python"))
train, val = data["train"].shuffle(seed=0), data["validation"]
val_fixed = val.shuffle(seed=1).select(range(args.eval_samples))
print(f"train {len(train)} | val {len(val)} (eval on fixed {args.eval_samples})")
if DIST and IS_MAIN:
    dist.barrier()

# --- the final training run pregenerated QA sidecar (--qa-data) ----------------------------------
# Records index the UNSHUFFLED train split (row_idx / stack_row_idxs), so all
# fetches for the pregen arms go through train_raw, not the seed-0 shuffle.
# Pool partition (deterministic, from the pinned record schema):
#   retr  = intent group I (descriptive retrieval; stack contexts)
#   span  = group B substring answers on solo contexts (span-targeted recon)
#   qapre = everything else (solo or stack context per the record)
# held_out records NEVER enter a pool — the sampled-held-out counter is the
# leak tripwire and must stay 0 forever (they ship for eval tooling only).
train_raw = data["train"]
QA = None
QA_SRC_EMA = {}   # per-source (ast/exec/luna/mcq) batch-loss EMA, rank-local
QA_STATS = {"leak": 0, "src_n": {}, "span_len": {}}
CUR_QA_SRC = None  # batch channel: builder -> micro_loss (sources of the qa batch)
if args.qa_data:
    import resource
    _rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _t_qa = time.time()
    # exact numeric shards only: glob qa_pairs_[0-9]* also matched e.g.
    # qa_pairs_0000_old.jsonl / qa_pairs_1.backup.jsonl (validation);
    # meta shards (qa_pairs_meta_*) stay excluded as before
    import re as _re
    _files = sorted(p for p in Path(args.qa_data).glob("qa_pairs_*.jsonl")
                    if _re.fullmatch(r"qa_pairs_\d+\.jsonl", p.name))
    assert _files, f"--qa-data: no qa_pairs_<digits>.jsonl in {args.qa_data}"
    _solo, _span, _conc, _qstack, _retr = [], [], [], {}, {}
    _held = 0
    _n_raw = len(train_raw)
    for _f in _files:
        for _l in open(_f):
            _r = json.loads(_l)
            if _r["held_out"]:
                _held += 1
                continue
            assert 0 <= _r["row_idx"] < _n_raw, \
                f"qa row_idx {_r['row_idx']} out of range for train split ({_n_raw})"
            _ctx = _r["context"]
            if _ctx["kind"] == "stack":
                _key = tuple(_ctx["stack_row_idxs"])
                assert all(0 <= i < _n_raw for i in _key), f"stack_row_idxs out of range: {_key}"
                (_retr if _r["intent"].startswith("I") else _qstack).setdefault(_key, []).append(_r)
            elif _r["intent"].startswith("B") and _r["answer_type"] == "substring":
                _span.append(_r)
            elif args.qapre_conc_frac > 0 and _r["intent"] in ("C1", "C3"):
                # conceptual (luna) tier gets its own stratum only when the
                # flag is on — at 0 the partition is byte-identical to legacy.
                # C2-luna is indistinguishable in the source record (source
                # mislabeled 'ast' pre-fix) and stays in the general pool.
                _conc.append(_r)
            else:
                _solo.append(_r)
    # stack pools: (sorted key list, dict) — sorted keys make the pure-fn group
    # draw deterministic and identical on every rank
    QA = {"qapre_solo": _solo, "span": _span, "qapre_conc": _conc,
          "qapre_stack": (sorted(_qstack), _qstack), "retr": (sorted(_retr), _retr),
          "qapre_stack_n": sum(len(v) for v in _qstack.values()),
          "ntok": train_raw["n_tokens"] if args.token_budget else None}
    assert not (args.qapre_conc_frac > 0 and not _conc), \
        "--qapre-conc-frac > 0 but no C1/C3 records in --qa-data"
    if FSDP_ON:
        # FSDP F4 invariant: build_stack_ctx encodes in row-chunks of 16,
        # so a pregen stack larger than 16 would make the comp() call count
        # depend on the RANK-LOCAL stack pick — a collective-schedule hang.
        # The final training run generator pins STACK_SIZE=8 (dataset/qa_extract.py); enforce.
        _mx = max((len(k) for k in (*_qstack, *_retr)), default=0)
        assert _mx <= 16, \
            f"--fsdp: pregen stack of {_mx} rows > build_stack_ctx chunk (16) — " \
            "rank-divergent comp() call counts would hang FSDP (DESIGN F4); " \
            "add a chunk-count sync before training on this qa-data"
    _rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _mb = (_rss1 - _rss0) / ((1 << 20) if sys.platform == "darwin" else (1 << 10))
    print(f"qa-data: {len(_files)} files -> qapre {len(_solo)} solo + "
          f"{QA['qapre_stack_n']} stacked ({len(_qstack)} stacks) + "
          f"conc {len(_conc)} (frac {args.qapre_conc_frac}) | "
          f"retr {sum(len(v) for v in _retr.values())} ({len(_retr)} stacks) | "
          f"span {len(_span)} | held-out excluded {_held} | "
          f"rss +{_mb:.0f}MB in {time.time() - _t_qa:.1f}s")
_NTOK = train["n_tokens"] if args.token_budget else None  # shuffled-view column for window trims
if args.token_budget:
    print(f"token-budget {args.token_budget}: realized eff batch is VARIABLE "
          f"(<= {WORLD}x{args.batch}x{args.grad_accum}); per-step realized rows/tokens "
          "logged in steps.jsonl ('tb')")

# --- token rarity table (drives loss weighting AND closed-book masking) -------
# Rarity over our own corpus: rare tokens are, almost by definition, the
# identifiers/literals — the high-entropy content the vectors keep fumbling.
rar_path = ROOT / "data" / "rarity.pt"
if DIST and not IS_MAIN:
    dist.barrier()  # rank 0 builds+saves rarity.pt if missing; wait, then load
if rar_path.exists():
    _r = torch.load(rar_path)
else:
    from collections import Counter

    cnt = Counter()
    for i in range(0, len(train), 4096):
        for seq in tok(train[i : i + 4096]["code"], add_special_tokens=False).input_ids:
            cnt.update(seq)
    freq = torch.zeros(len(tok))
    for t, c in cnt.items():
        freq[t] = c
    total = freq.sum()
    surp = -(freq / total).clamp_min(1e-12).log()
    surp = surp.clamp(max=surp[freq > 0].max())          # unseen tokens = max seen rarity
    rarity = (surp - surp.min()) / (surp.max() - surp.min())
    _r = {"rarity": rarity, "is_rare": freq < total * 1e-5}
    torch.save(_r, rar_path)
    print(f"rarity table built: {int(_r['is_rare'].sum())} rare token types")
if DIST and IS_MAIN:
    dist.barrier()
rarity_w = _r["rarity"].to(device)
is_rare = _r["is_rare"].to(device)

# --- contextual surprisal column ( --surprisal-weight) -------------------
# Keyed by md5(code): the column is built over the UNSHUFFLED train split by
# the surprisal-data builder; the trainer sees a seed-0 shuffle of the same rows.
SURP = None
SURP_CLAMP = 16.0
if args.surprisal_weight > 0:
    assert args.rare_weight == 0, "--surprisal-weight replaces --rare-weight (pick one price list)"
    import hashlib

    _sc = torch.load(ROOT / "data" / "surprisal_train.pt")
    SURP_CLAMP = float(_sc["clamp"])
    _codes = data["train"]["code"]
    if "md5" in _sc:
        # keyed column (validation fix): order-independent, rebuild-proof
        SURP = dict(zip(_sc["md5"], _sc["surp"]))
        print(f"surprisal column loaded (md5-keyed): {len(SURP)} rows, clamp {SURP_CLAMP}")
    else:
        # legacy positional column: pairing by index is only valid if the dataset
        # is byte-identical to the one the column was built over. Verify: each
        # row's tensor length must equal its paired code's token count — a
        # reorder/rebuild breaks that for nearly every row. Hard abort beats
        # silently training with wrong weights under valid-looking hashes.
        if len(_codes) != len(_sc["surp"]):
            print(f"WARNING: surprisal column PARTIAL ({len(_sc['surp'])}/{len(_codes)} rows) — "
                  "unmatched rows train UNWEIGHTED (watch surp-hits; fine for smokes, "
                  "build the full column for real runs)")
        _n_chk = min(200, len(_sc["surp"]))
        _idx = range(0, len(_sc["surp"]), max(1, len(_sc["surp"]) // _n_chk))
        _bad = sum(len(_sc["surp"][i]) != len(tok(_codes[i], add_special_tokens=False).input_ids)
                   for i in _idx)
        assert _bad <= len(list(_idx)) * 0.01, \
            f"surprisal column MISALIGNED with dataset ({_bad}/{len(list(_idx))} sampled rows " \
            "have wrong tensor length) — dataset was rebuilt/reordered; regenerate the column " \
            "with the surprisal-data builder (new versions save md5 keys and are immune)"
        SURP = {hashlib.md5(c.encode()).hexdigest(): t
                for c, t in zip(_codes[: len(_sc["surp"])], _sc["surp"])}
        print(f"surprisal column loaded (positional, alignment-verified "
              f"{len(list(_idx))} rows): {len(SURP)} rows, clamp {SURP_CLAMP}")


SURP_HITS = [0, 0]  # [found, missed] — a silent all-miss would make the flag a no-op


def surp_row(code):
    """per-token surprisal for one function, or None if not in the column"""
    import hashlib

    t = SURP.get(hashlib.md5(code.encode()).hexdigest()) if SURP is not None else None
    SURP_HITS[0 if t is not None else 1] += 1
    return t.to(device).float() if t is not None else None


# per-batch weight map (B, L) aligned with labels; set by the recon-family batch
# builders, consumed-and-cleared by loss_on. A module global instead of a fifth
# tuple slot everywhere: single-threaded loop, build->loss is always adjacent.
CUR_WMAP = None

if args.pooling == "latent":
    # surprisal prior for the allocation pooler: rare tokens get an attention
    # bonus from step 0. Deterministic from the corpus, so overwriting after a
    # warm start is idempotent.
    comp.projector.latent_pooler.rarity_tbl.copy_(rarity_w)
    if args.span_scaffold > 0:
        comp.projector.latent_pooler.keep_attn = True
        print(f"span scaffold ON: lambda {args.span_scaffold} -> 0 over "
              f"{args.scaffold_steps} steps (AST teaches early, then dissolves)")


# --- prose detection (docstrings + comments) -----------------------------------
# Evaluation showed most reconstruction failures are docstring PARAPHRASE,
# not code errors. Prose wording is fungible for the use case; these spans get
# --prose-weight in the loss and are exempt from rarity boosts / masking.
import re

_TRIPLE = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')
_COMMENT = re.compile(r"#[^\n]*")
_WS = re.compile(r"[ \t]+")
_BLANK = re.compile(r"\n\s*\n+")


def prose_spans(code: str):
    return [(m.start(), m.end()) for rx in (_TRIPLE, _COMMENT) for m in rx.finditer(code)]


def strip_prose_text(s: str) -> str:
    """Same normalization as the evaluation utilities: prose removed, whitespace collapsed."""
    s = _TRIPLE.sub('""', s)
    s = _COMMENT.sub("", s)
    s = _WS.sub(" ", s)
    s = _BLANK.sub("\n", s)
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def prose_token_mask(codes, offsets):
    """(B, C) bool: token's char span overlaps a docstring/comment span."""
    B, C = offsets.shape[:2]
    mask = torch.zeros(B, C, dtype=torch.bool)
    for i, code in enumerate(codes):
        spans = prose_spans(code)
        if not spans:
            continue
        for j in range(C):
            a, b = int(offsets[i, j, 0]), int(offsets[i, j, 1])
            if a == b:
                continue  # padding/special
            if any(a < e and b > s for s, e in spans):
                mask[i, j] = True
    return mask


# Fixed shape buckets: MPS compiles kernels per tensor shape, so unbounded
# shape variety (every batch a different padded length) makes step time creep
# upward. Four buckets -> four shape families, stable speed.
BUCKETS = (128, 256, 384, 512)
if args.long_buckets:  # the final training run: the corpus has a 512-4096 long tail (10% of rows)
    BUCKETS = (128, 256, 384, 512, 1024, 2048, 4096)

# lever F telemetry (--pad-to-max, training builds only): realized encoder pad
# widths since the last steps.jsonl write — [n, w_min, w_max, not-mult-of-16].
# Host-side ints from shapes, zero device syncs; nm16 is the no-op tripwire.
PAD_MULT = 16
PAD_STATS = [0, 0, 0, 0]


def _pad_target(C, stats=True):
    """padded encoder width for a batch whose longest row is C tokens: the
    shape bucket (legacy) or, under --pad-to-max, C rounded up to 16. The
    bucket CEILING stays enforced either way (corpus contract: rows over
    BUCKETS[-1] need --long-buckets). stats=False for secondary callers that
    mirror an already-counted width (the prose pad)."""
    if C > BUCKETS[-1]:
        raise ValueError(f"row of {C} tokens exceeds bucket ceiling {BUCKETS[-1]} "
                         "— this corpus needs --long-buckets")
    if args.pad_to_max:
        w = max(PAD_MULT, -(-C // PAD_MULT) * PAD_MULT)
        if stats and comp.training:
            PAD_STATS[0] += 1
            PAD_STATS[1] = w if PAD_STATS[1] == 0 else min(PAD_STATS[1], w)
            PAD_STATS[2] = max(PAD_STATS[2], w)
            PAD_STATS[3] += int(w % PAD_MULT != 0)
        return w
    return next(x for x in BUCKETS if x >= C)


def _pad_len(L):
    """--pad-to-max: assembled DECODER lengths round up to 16 too (they are
    already max-in-batch-tight; the extra tail is dec_mask 0 / labels -100 —
    masked everywhere, mathematically inert). Identity when the flag is off."""
    return -(-L // PAD_MULT) * PAD_MULT if args.pad_to_max else L


def to_bucket(ids, mask):
    C = ids.shape[1]
    b = _pad_target(C)
    pad = b - C
    if pad:
        ids = torch.nn.functional.pad(ids, (0, pad), value=tok.pad_token_id)
        mask = torch.nn.functional.pad(mask, (0, pad))
    return ids, mask


# --- lever D (--build-prefetch): main-thread cache seam ------------------------
# The worker thread (defined near _dstream, which it needs) fills a per-micro
# cache of CPU-side build products for micro k+1 while the GPU runs micro k.
# These wrappers are the ONLY way builders consume it: on a hit they return
# the worker's product (bit-identical by construction — the key embeds the
# full input), on a miss they fall through to the exact legacy call. Flag off
# => _PF_ACTIVE stays None => pure pass-through. Speed-only layer: a stale or
# empty cache can never change realized batches, only the hit rate (tripwire:
# pfetch_r0.tok_hit == 0 with the flag on = dead cache).
_PF_ACTIVE = None  # current micro's prefetch cache dict (main thread only)
PF_STATS = {"rows_hit": 0, "rows_miss": 0, "tok_hit": 0, "tok_miss": 0,
            "prep_err": 0, "wait_ms": 0.0}  # wait_ms: EMA of main-thread stall


def _ctok_batch(texts, offsets=False):
    """tok(texts, return_tensors='pt', padding=True, add_special_tokens=False
    [, return_offsets_mapping=True]) — the builders' one batch-tokenize shape."""
    if _PF_ACTIVE is not None:
        e = _PF_ACTIVE.get(("tb", tuple(texts), offsets))
        if e is not None:
            PF_STATS["tok_hit"] += 1
            return e
        PF_STATS["tok_miss"] += 1
    kw = {"return_tensors": "pt", "padding": True, "add_special_tokens": False}
    if offsets:
        kw["return_offsets_mapping"] = True
    return tok(texts, **kw)


def _ctok_ids(text):
    """tok(text, add_special_tokens=False).input_ids — the single-string shape."""
    if _PF_ACTIVE is not None:
        e = _PF_ACTIVE.get(("ti", text))
        if e is not None:
            PF_STATS["tok_hit"] += 1
            return e
        PF_STATS["tok_miss"] += 1
    return tok(text, add_special_tokens=False).input_ids


def _cprose(codes, offsets):
    """prose_token_mask, cache-served (pure CPU python over the offsets)."""
    if _PF_ACTIVE is not None:
        e = _PF_ACTIVE.get(("pr", tuple(codes)))
        if e is not None:
            return e
    return prose_token_mask(codes, offsets)


def _cfacts(code):
    """fn_facts, cache-served (pure AST parse)."""
    if _PF_ACTIVE is not None and ("fa", code) in _PF_ACTIVE:
        return _PF_ACTIVE[("fa", code)]
    return fn_facts(code)


def _cdoc(code):
    """extract_docstring, cache-served (regex + tokenize/decode round-trip)."""
    if _PF_ACTIVE is not None and ("dc", code) in _PF_ACTIVE:
        return _PF_ACTIVE[("dc", code)]
    return extract_docstring(code)


def _craw(idxs):
    """train_raw[idxs] (pregen-arm context rows), cache-served."""
    if _PF_ACTIVE is not None:
        e = _PF_ACTIVE.get(("rr", tuple(idxs)))
        if e is not None:
            PF_STATS["rows_hit"] += 1
            return e
        PF_STATS["rows_miss"] += 1
    return train_raw[list(idxs)]


# constant-prompt tensor cache (syncs-kill bundle, unconditional & exact):
# assemble/stack builders re-upload the same small prompt id-lists every micro
# (task markers, headers, qa prompts). Same ints -> same device tensor; bounded.
_PROMPT_T = {}


def _prompt_tensor(ids):
    k = tuple(ids)
    t = _PROMPT_T.get(k)
    if t is None:
        if len(_PROMPT_T) > 4096:
            _PROMPT_T.clear()
        t = torch.tensor(ids, device=device)
        _PROMPT_T[k] = t
    return t


def build_anchor_masks(enc_ids, enc_att, prose):
    """(B, T) bool: per sample, top --anchor-frac of real tokens by rarity,
    prose excluded. The trainer-side (batched) twin of anchors.anchor_positions."""
    r = rarity_w[enc_ids].masked_fill(~enc_att.bool(), -1.0)
    if prose is not None:
        r = r.masked_fill(prose, -1.0)
    out = torch.zeros_like(enc_att, dtype=torch.bool)
    n_tok_l = enc_att.sum(dim=1).tolist()  # one host sync, not one per row
    for i in range(enc_ids.shape[0]):
        k = min(int(n_tok_l[i] * args.anchor_frac), int((r[i] >= 0).sum()))
        if k > 0:
            out[i, r[i].topk(k).indices] = True
    return out


# ---- scaffold channel --------------------------------------------------
# Set by make_batch (recon task only), consumed-and-cleared by the main loop.
# Holds the already-computed scaffold CE loss (graph attached) so later comp()
# calls (distractor, other tasks) can't clobber the attention it was built on.
CUR_SCAF_LOSS = None
CUR_PSHORT = None  # (B,) bool: sample's docstring is short (<200 chars)

if args.span_scaffold > 0 or args.prose_weight_short is not None:
    from compressor.spans import segment as _seg_spans, span_masses as _span_masses


def scaffold_target(codes, offsets, T):
    """(B,T) proposed per-token attention mass: each span's surprisal mass
    (char mass if no column) spread uniformly over its tokens, normalized."""
    tgt = torch.zeros(len(codes), T)
    short = torch.zeros(len(codes), dtype=torch.bool)
    for i, code in enumerate(codes):
        spans = _seg_spans(code)
        doc_len = sum(e - s for k, s, e in spans if k == "docstring")
        short[i] = 0 < doc_len < 200
        sr = surp_row(code) if SURP is not None else None
        masses = _span_masses(code, spans, None, None) if sr is None else None
        offs = offsets[i]
        # token -> span assignment by start char
        span_tok = [[] for _ in spans]
        for t_idx, (cs, ce) in enumerate(offs):
            if ce == cs:
                continue  # padding/special
            for si, (k, s, e) in enumerate(spans):
                if s <= cs < e:
                    span_tok[si].append(t_idx)
                    break
        if masses is None:
            m = []
            for si, (k, s, e) in enumerate(spans):
                idxs = span_tok[si]
                m.append(sum(float(sr[t]) for t in idxs if t < len(sr)) if idxs else 0.0)
            tot = sum(m)
            masses = [x / tot for x in m] if tot > 0 else [1.0 / len(spans)] * len(spans)
        for si, idxs in enumerate(span_tok):
            if idxs:
                w = masses[si] / len(idxs)
                for t_idx in idxs:
                    if t_idx < T:
                        tgt[i, t_idx] = w
    tgt = tgt / tgt.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return tgt, short


def make_batch(rows, mask_p=0.0):
    """rows -> (dec_embeds, dec_mask, labels, prose_map), padded to a fixed bucket.

    mask_p > 0 (training only): rare tokens in the visible history are blanked
    with this probability, so their spelling can only come from the vectors.
    prose_map marks label positions inside docstrings/comments (None if
    --prose-weight is 1.0, keeping legacy behavior bit-identical).
    --anchor-frac > 0: each group's vector is followed by its anchor tokens'
    raw embeddings in the context prefix (router stage 0).
    """
    codes = rows["code"]
    want_prose = args.prose_weight != 1.0 or args.anchor_frac > 0
    want_scaf = args.span_scaffold > 0 or args.prose_weight_short is not None
    enc = _ctok_batch(codes, offsets=want_prose or want_scaf)
    prose = _cprose(codes, enc.offset_mapping).to(device) if want_prose else None
    enc_ids, enc_att = enc.input_ids.to(device), enc.attention_mask.to(device)
    if want_prose:
        C = enc_ids.shape[1]
        b = _pad_target(C, stats=False)  # must match to_bucket's width below
        prose = torch.nn.functional.pad(prose, (0, b - C))
    enc_ids, enc_att = to_bucket(enc_ids, enc_att)
    vecs = comp(enc_ids, enc_att)  # (B, Gmax, H)

    global CUR_SCAF_LOSS, CUR_PSHORT
    CUR_SCAF_LOSS = CUR_PSHORT = None
    if want_scaf:
        tgt, short = scaffold_target(codes, enc.offset_mapping.tolist(), enc_ids.shape[1])
        CUR_PSHORT = short.to(device)
        if args.span_scaffold > 0 and comp.training and args.pooling == "latent":
            pa = comp.projector.latent_pooler.last_attn        # (B,H,G,T) with grad
            mass = pa.mean(1).sum(1) * enc_att                 # (B,T), padding zeroed
            mass = mass / mass.sum(1, keepdim=True).clamp_min(1e-8)
            t = tgt.to(device)
            # CE(proposal, actual): pushes allocation toward the AST+surprisal
            # proposal; weight annealed to zero in the main loop
            CUR_SCAF_LOSS = -(t * (mass + 1e-8).log()).sum(1).mean()

    pf = comp.pooling_factor
    n_tok = enc_att.sum(dim=1)                                 # real tokens/sample
    n_grp = torch.ceil(n_tok.float() / pf).long()              # real vec groups/sample
    anch = build_anchor_masks(enc_ids, enc_att, prose) if args.anchor_frac > 0 else None

    B, Gmax, H = vecs.shape
    Cmax = enc_ids.shape[1]
    bp = comp.boundary_pair()
    n_extra = (2 if bp is not None else 0) + (1 if args.eos else 0)
    n_anch = anch.sum(dim=1) if anch is not None else torch.zeros(B, dtype=torch.long, device=device)
    L = _pad_len((int((n_grp + n_anch + n_tok).max()) if anch is not None else Gmax + Cmax) + n_extra)
    dec_in = torch.zeros(B, L, H, dtype=embed_table.weight.dtype, device=device)
    dec_mask = torch.zeros(B, L, dtype=torch.long, device=device)
    labels = torch.full((B, L), -100, dtype=torch.long, device=device)
    pmap = torch.zeros(B, L, dtype=torch.bool, device=device) if want_prose else None
    wmap = torch.ones(B, L, dtype=torch.float32, device=device) if SURP is not None else None

    code_embeds = embed_table(enc_ids)
    anchor_embeds = code_embeds  # unblanked: anchors must stay copyable
    if mask_p > 0:
        blank = is_rare[enc_ids] & enc_att.bool() & (torch.rand(enc_ids.shape, device=device) < mask_p)
        if want_prose:
            blank &= ~prose  # don't force vectors to memorize prose spellings
        code_embeds = code_embeds.masked_fill(blank.unsqueeze(-1), 0.0)
    # syncs-kill (exact): one .tolist() per micro instead of 2 host syncs/row
    n_grp_l, n_tok_l = n_grp.tolist(), n_tok.tolist()
    eos_emb = embed_table.weight[tok.eos_token_id] if args.eos else None
    for i in range(B):
        g, c = n_grp_l[i], n_tok_l[i]
        if anch is None:
            pos = 0
            if bp is not None:
                dec_in[i, 0] = bp[0][0, 0]
                pos = 1
            dec_in[i, pos : pos + g] = vecs[i, :g]
            pos += g
            if bp is not None:
                dec_in[i, pos] = bp[1][0, 0]
                pos += 1
        else:
            pos = 0
            anch_i = anch[i].nonzero(as_tuple=True)[0].tolist()
            ai = 0
            for gi in range(g):
                dec_in[i, pos] = vecs[i, gi]
                pos += 1
                hi = (gi + 1) * pf
                while ai < len(anch_i) and anch_i[ai] < hi:
                    dec_in[i, pos] = anchor_embeds[i, anch_i[ai]]
                    pos += 1
                    ai += 1
        dec_in[i, pos : pos + c] = code_embeds[i, :c]
        labels[i, pos : pos + c] = enc_ids[i, :c]
        if want_prose:
            pmap[i, pos : pos + c] = prose[i, :c]
        if SURP is not None:
            sr = surp_row(codes[i])
            if sr is not None and len(sr) >= c:
                wmap[i, pos : pos + c] = 1.0 + args.surprisal_weight * (sr[:c] / SURP_CLAMP)
        if args.eos:
            dec_in[i, pos + c] = eos_emb
            labels[i, pos + c] = tok.eos_token_id
            c += 1
        dec_mask[i, : pos + c] = 1
    global CUR_WMAP
    CUR_WMAP = wmap
    return dec_in, dec_mask, labels, pmap


from torch.utils.checkpoint import checkpoint as _ckpt


MARGIN_K = 0.0  # ramped copy of --margin-weight, set per step in the main loop
# margin telemetry (the logging policy validation: anything loggable gets logged —
# we flew a whole fine-tune blind on THE leading indicator, whether near-ties
# widen): [sum_margin, n, n_negative, n_below_tau], reset at each step line.
MARGIN_STATS = [0.0, 0, 0, 0]
# --telemetry-batched: device-resident twins of MARGIN_STATS / TB_STATS. The
# hot path only launches device ops (no .item()/float()/int() host syncs —
# profiled at up to ~24 syncs/micro on 8xH100); the Python lists are populated
# from these ONLY at the existing logging boundaries (%10 tb / %50 margin), so
# keys, semantics and cadence are unchanged. Counts ride int64 (exact); the
# margin sum rides fp32 (device accumulation order may differ from the legacy
# per-chunk python-float sum — tolerated, documented). Telemetry stays outside
# the checkpointed _ce_slice either way (recompute would double-count).
TELEM_BATCH = args.telemetry_batched
if TELEM_BATCH:
    MARGIN_ACC_F = torch.zeros((), dtype=torch.float32, device=device)  # sum_margin
    MARGIN_ACC_I = torch.zeros(3, dtype=torch.int64, device=device)     # n, n_neg, n<tau
RANK_K = 0.0  # ramped copy of --rank-loss, set per step in the main loop
# rank-hinge telemetry, reset at each step line:
# [nonprose_n, hinge_active_n, hinge_sum, margin_sum_nonprose,
#  conf_margin_sum, conf_n]  (conf = margin >= m*, the earlier configuration narrowing detector)
RANK_STATS = [0, 0, 0.0, 0.0, 0.0, 0]
if args.rank_loss > 0:
    assert args.chunked_loss, "--rank-loss lives in the chunked-loss path only"
    assert args.prose_weight != 1.0, \
        "--rank-loss needs the prose mask active (set --prose-weight < 1)"
if args.rank_active_norm:
    assert args.rank_loss > 0, "--rank-active-norm modifies the hinge; needs --rank-loss"
if args.hard_mine > 0:
    assert args.rank_loss > 0, "--hard-mine attributes hinge-active ties; needs --rank-loss"
# hard-mine channel + pool. CUR_ROWIDS: main loop -> loss_on, dataset row ids of
# the current RECON batch (attribution only; consumed-and-cleared like CUR_WMAP).
# HARD_EMA: rowid -> EMA of that row's hinge-active token count.
# HARD_STATS (reset at each step line): [swapped_n, attributed_batches, attributed_act_sum]
CUR_ROWIDS = None
HARD_EMA = {}
HARD_STATS = [0, 0, 0.0]
RANK_GATE = True  # with --rank-recon-only: main loop sets per task, loss_on reads


def _ce_slice(h_chunk, lb_chunk):
    lg = lm_head(h_chunk).float()
    ce = torch.nn.functional.cross_entropy(lg, lb_chunk, reduction="none")
    # gold-rival margin. an earlier configuration consumed it no-grad as a CE WEIGHT (dead family);
    # the ranking hinge needs gradient THROUGH the gap, so it stays in the
    # graph — still just topk(2)+gather, no 151k-wide clone. Consumers that
    # only weight or log must .detach() it.
    top2 = lg.topk(2, dim=-1)
    true_lg = lg.gather(-1, lb_chunk.unsqueeze(-1)).squeeze(-1)
    rival = torch.where(top2.indices[:, 0] == lb_chunk,
                        top2.values[:, 1], top2.values[:, 0])
    margin = true_lg - rival
    return ce, margin


def extract_docstring(code, max_toks=96, tokenizer=None):
    """First triple-quoted string's contents, whitespace-collapsed, or None.
    tokenizer: the --build-prefetch worker passes its OWN instance (fast
    tokenizers race on concurrent padding-state mutation); default = tok."""
    t_ = tokenizer or tok
    m = _TRIPLE.search(code)
    if not m:
        return None
    doc = _WS.sub(" ", m.group(0).strip('"\'')).strip()
    if len(doc) < 8:
        return None
    ids = t_(doc, add_special_tokens=False).input_ids[:max_toks]
    return t_.decode(ids)


# task markers (--task-markers): short no-loss prompts that make cont/desc
# contexts structurally distinct from recon — the decoder should never have to
# guess which task it's in (the CoverageConfig mode-confusion, validation)
MARK_CONT = tok("\nContinue:\n", add_special_tokens=False).input_ids
MARK_DESC = tok("\nDescribe:\n", add_special_tokens=False).input_ids


def assemble(vecs, n_grp, tgt_ids, tgt_att, pmap_src=None, prefix_ids=None, add_eos=False,
             use_boundary=True, wmap_src=None):
    """[<block>][vector groups][</block>][prefix text][target embeds] -> batch tuple.
    prefix_ids: per-sample list of id-lists inserted between block and target with
    NO loss (question text for the qa task). add_eos: EOS appended to each target
    (with loss) so the model learns to stop. wmap_src: per-sample surprisal rows
    aligned with tgt positions (--surprisal-weight); sets the CUR_WMAP channel."""
    global CUR_WMAP
    B, Gmax, H = vecs.shape
    bp = comp.boundary_pair() if use_boundary else None
    nb = 2 if bp is not None else 0
    Pmax = max((len(q) for q in prefix_ids), default=0) if prefix_ids is not None else 0
    L = _pad_len(Gmax + nb + Pmax + tgt_ids.shape[1] + (1 if add_eos else 0))
    dec_in = torch.zeros(B, L, H, dtype=embed_table.weight.dtype, device=device)
    dec_mask = torch.zeros(B, L, dtype=torch.long, device=device)
    labels = torch.full((B, L), -100, dtype=torch.long, device=device)
    pmap = torch.zeros(B, L, dtype=torch.bool, device=device) if pmap_src is not None else None
    wmap = torch.ones(B, L, dtype=torch.float32, device=device) if wmap_src is not None else None
    tgt_emb = embed_table(tgt_ids)
    # syncs-kill (exact): one .tolist() per micro instead of a host sync per row
    n_grp_l, n_tgt_l = n_grp.tolist(), tgt_att.sum(dim=1).tolist()
    eos_emb = embed_table.weight[tok.eos_token_id] if add_eos else None
    for i in range(B):
        g, c = n_grp_l[i], n_tgt_l[i]
        pos = 0
        if bp is not None:
            dec_in[i, 0] = bp[0][0, 0]
            pos = 1
        dec_in[i, pos : pos + g] = vecs[i, :g]
        pos += g
        if bp is not None:
            dec_in[i, pos] = bp[1][0, 0]
            pos += 1
        if prefix_ids is not None and prefix_ids[i]:
            q = _prompt_tensor(prefix_ids[i])
            dec_in[i, pos : pos + len(q)] = embed_table(q)
            pos += len(q)
        dec_in[i, pos : pos + c] = tgt_emb[i, :c]
        labels[i, pos : pos + c] = tgt_ids[i, :c]
        if pmap is not None:
            pmap[i, pos : pos + c] = pmap_src[i, :c]
        if wmap is not None and wmap_src[i] is not None and len(wmap_src[i]) >= c:
            wmap[i, pos : pos + c] = 1.0 + args.surprisal_weight * (wmap_src[i][:c] / SURP_CLAMP)
        if add_eos and c > 0:
            dec_in[i, pos + c] = eos_emb
            labels[i, pos + c] = tok.eos_token_id
            c += 1
        dec_mask[i, : pos + c] = 1
    CUR_WMAP = wmap
    return dec_in, dec_mask, labels, pmap


def make_batch_cont(rows):
    """Continuation: vectors of the FIRST half must support predicting the
    second half — vectors have to be useful, not just replayable."""
    want_prose = args.prose_weight != 1.0
    enc = _ctok_batch(rows["code"], offsets=want_prose)
    prose = _cprose(rows["code"], enc.offset_mapping).to(device) if want_prose else None
    ids, att = enc.input_ids.to(device), enc.attention_mask.to(device)
    B = ids.shape[0]
    n = att.sum(dim=1)
    mid = (n // 2).clamp(min=16)
    idx = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    first_att = ((idx < mid.unsqueeze(1)) & att.bool()).long()
    src_ids, src_att = to_bucket(ids * first_att, first_att)
    vecs = comp(src_ids, src_att)
    n_grp = torch.ceil(first_att.sum(dim=1).float() / comp.pooling_factor).long()

    # syncs-kill (exact): batch the per-row mid/n host reads into one .tolist()
    mid_l, n_l = mid.tolist(), n.tolist()
    Tmax = max(b - a for a, b in zip(mid_l, n_l))
    tgt_ids = torch.zeros(B, Tmax, dtype=torch.long, device=device)
    tgt_att = torch.zeros(B, Tmax, dtype=torch.long, device=device)
    tgt_prose = torch.zeros(B, Tmax, dtype=torch.bool, device=device) if want_prose else None
    for i in range(B):
        a, b = mid_l[i], n_l[i]
        tgt_ids[i, : b - a] = ids[i, a:b]
        tgt_att[i, : b - a] = 1
        if want_prose:
            tgt_prose[i, : b - a] = prose[i, a:b]
    mark = [MARK_CONT] * B if args.task_markers else None
    wsrc = None
    if SURP is not None:
        # second-half slice of each function's surprisal row
        wsrc = []
        for i in range(B):
            sr = surp_row(rows["code"][i])
            a = mid_l[i]
            wsrc.append(sr[a:] if sr is not None and len(sr) > a else None)
    return assemble(vecs, n_grp, tgt_ids, tgt_att, tgt_prose, prefix_ids=mark, wmap_src=wsrc)


def make_batch_desc(rows):
    """Describe: compress prose-stripped code, predict its docstring — a
    'did you understand it' objective. Rows without docstrings get no loss."""
    docs = [_cdoc(c) for c in rows["code"]]
    stripped = [strip_prose_text(c) for c in rows["code"]]
    enc = _ctok_batch(stripped)
    enc_ids, enc_att = to_bucket(enc.input_ids.to(device), enc.attention_mask.to(device))
    vecs = comp(enc_ids, enc_att)
    n_grp = torch.ceil(enc_att.sum(dim=1).float() / comp.pooling_factor).long()
    tgt = _ctok_batch([d or "" for d in docs])
    tgt_ids, tgt_att = tgt.input_ids.to(device), tgt.attention_mask.to(device)
    for i, d in enumerate(docs):
        if d is None:
            tgt_att[i] = 0  # no docstring -> no loss for this row
    mark = [MARK_DESC] * len(docs) if args.task_markers else None
    return assemble(vecs, n_grp, tgt_ids, tgt_att, None, prefix_ids=mark)  # no prose-weight: prose IS the target


def make_batch_plain(rows):
    """Anti-forgetting rehearsal: ordinary LM loss on raw code, NO vectors.
    Mirrors the decoder's native training so LoRA can't drift off plain text.
    Train with weighted=False — rehearsal must replicate the original objective,
    not our rarity-boosted one."""
    enc = _ctok_batch(rows["code"])
    ids, att = to_bucket(enc.input_ids.to(device), enc.attention_mask.to(device))
    labels = ids.masked_fill(att == 0, -100)
    global CUR_WMAP
    CUR_WMAP = None  # rehearsal is unweighted; don't leak a stale surprisal map
    return embed_table(ids), att, labels, None


_DEF = re.compile(r"(?:async\s+)?def\s+(\w+)")


def build_stack_ctx(codes, names):
    """Shared labeled context: [### function: name\\n][<block>][vecs][</block>] per fn.
    Returns (ctx (Lc,H), enc_ids, n_tok) for downstream target construction."""
    enc = _ctok_batch(codes)
    enc_ids, enc_att = to_bucket(enc.input_ids.to(device), enc.attention_mask.to(device))
    # encoder in row-chunks: N=80 stacks pad to an 80x448 batch whose transient
    # forward peak OOM'd 95GB three times (validation); token budgets can't help
    # because padding, not content, sets the effective batch. Chunks join one
    # autograd graph; enc-grad-ckpt bounds the stored activations either way.
    _CH = 16
    if enc_ids.shape[0] > _CH:
        # : chunks join ONE autograd graph — announce the whole-micro
        # encoder footprint so the ckpt-threshold gate prices the total, not
        # each 16-row slice (which could each duck under N while their stored
        # activations coexist). No-op when --ckpt-token-threshold is 0.
        CKPT_ANN["enc"] = enc_ids.shape[0] * enc_ids.shape[1]
        try:
            vecs = torch.cat([comp(enc_ids[s:s + _CH], enc_att[s:s + _CH])
                              for s in range(0, enc_ids.shape[0], _CH)], dim=0)
        finally:
            CKPT_ANN["enc"] = 0
    else:
        vecs = comp(enc_ids, enc_att)
    n_tok = enc_att.sum(dim=1)
    n_grp_l = torch.ceil(n_tok.float() / comp.pooling_factor).long().tolist()  # one sync
    edt = embed_table.weight.dtype
    bp = comp.boundary_pair()
    parts = []
    for i, name in enumerate(names):
        hdr = _prompt_tensor(_ctok_ids(f"### function: {name}\n"))
        parts.append(embed_table(hdr))
        if bp is not None:
            parts.append(bp[0][0])
        parts.append(vecs[i, : n_grp_l[i]].to(edt))
        if bp is not None:
            parts.append(bp[1][0])
    return torch.cat(parts, dim=0), enc_ids, n_tok


def stack_members(rows, rng):
    """N in [2, --stack-n-max] functions: the batch's rows plus extra train rows
    if N exceeds the batch. Sample count shrinks as N grows (activation bound)."""
    N = rng.randint(2, args.stack_n_max)
    codes = list(rows["code"])
    while len(codes) < N:
        codes.append(train[rng.randrange(len(train))]["code"])
    codes = codes[:N]
    if args.stack_token_budget:
        kept, total = [], 0
        for c in codes:
            t = len(_ctok_ids(c))
            if kept and len(kept) >= 2 and total + t > args.stack_token_budget:
                continue  # skip long ones, keep filling with shorter draws
            kept.append(c)
            total += t
        codes = kept
        N = len(codes)
    if FSDP_SYNC:
        # FSDP F4: the N draw AND the token-budget trim are rank-local, and
        # build_stack_ctx issues ceil(N/16) comp() calls — rank-divergent
        # counts hang the collective schedule (measured). Truncate every rank
        # to the cross-rank MIN; count_div_* telemetry prices the loss.
        N = _sync_fwd_count(N)
        codes = codes[:N]
    names = [(m.group(1) if (m := _DEF.search(c)) else f"fn{i}") for i, c in enumerate(codes)]
    n_samples = max(2, min(args.batch, 40 // max(N, 1)))
    targets = rng.sample(range(N), min(n_samples, N))
    return codes, names, targets


def make_batch_stack(rows, srng):
    """In-flow retrieval: N functions in one labeled context; each sample
    reproduces the one asked for by name. srng: stack content stream (legacy
    stack_rng; rank-local pure-fn draw under --distributed)."""
    codes, names, targets = stack_members(rows, srng)
    ctx, enc_ids, n_tok = build_stack_ctx(codes, names)
    Lc = ctx.shape[0]
    prompts = [_ctok_ids(f"\nReproduce the function `{names[t]}`:\n") for t in targets]
    B = len(targets)
    L = _pad_len(Lc + max(len(q) for q in prompts) + enc_ids.shape[1] + (1 if args.eos else 0))
    dec_in = torch.zeros(B, L, ctx.shape[1], dtype=embed_table.weight.dtype, device=device)
    dec_mask = torch.zeros(B, L, dtype=torch.long, device=device)
    labels = torch.full((B, L), -100, dtype=torch.long, device=device)
    wmap = torch.ones(B, L, dtype=torch.float32, device=device) if SURP is not None else None
    code_emb = embed_table(enc_ids)
    n_tok_l = n_tok.tolist()  # syncs-kill (exact): one host sync per micro
    eos_emb = embed_table.weight[tok.eos_token_id] if args.eos else None
    for j, t in enumerate(targets):
        q = _prompt_tensor(prompts[j])
        pl, c = q.shape[0], n_tok_l[t]
        dec_in[j, :Lc] = ctx
        dec_in[j, Lc : Lc + pl] = embed_table(q)
        dec_in[j, Lc + pl : Lc + pl + c] = code_emb[t, :c]
        labels[j, Lc + pl : Lc + pl + c] = enc_ids[t, :c]
        if wmap is not None:
            sr = surp_row(codes[t])
            if sr is not None and len(sr) >= c:
                wmap[j, Lc + pl : Lc + pl + c] = 1.0 + args.surprisal_weight * (sr[:c] / SURP_CLAMP)
        if args.eos:
            dec_in[j, Lc + pl + c] = eos_emb
            labels[j, Lc + pl + c] = tok.eos_token_id
            c += 1
        dec_mask[j, : Lc + pl + c] = 1
    global CUR_WMAP
    CUR_WMAP = wmap
    return dec_in, dec_mask, labels, None


def _stack_qa_assemble(ctx, prompts, a_ids, a_att, add_eos):
    """[shared stack ctx][prompt_i][answer_i(+EOS)] per sample, loss on the
    answer(+EOS) only — the qa stack variant's assembly, shared verbatim with
    the final training run pregen qapre-stack/retr arms."""
    Lc = ctx.shape[0]
    B = len(prompts)
    L = _pad_len(Lc + max(len(p_) for p_ in prompts) + a_ids.shape[1] + (1 if add_eos else 0))
    dec_in = torch.zeros(B, L, ctx.shape[1], dtype=embed_table.weight.dtype, device=device)
    dec_mask = torch.zeros(B, L, dtype=torch.long, device=device)
    labels = torch.full((B, L), -100, dtype=torch.long, device=device)
    a_emb = embed_table(a_ids)
    n_a_l = a_att.sum(dim=1).tolist()  # syncs-kill (exact): one host sync
    eos_emb = embed_table.weight[tok.eos_token_id] if add_eos else None
    for i in range(B):
        q = _prompt_tensor(prompts[i])
        pl, c = q.shape[0], n_a_l[i]
        dec_in[i, :Lc] = ctx
        dec_in[i, Lc : Lc + pl] = embed_table(q)
        dec_in[i, Lc + pl : Lc + pl + c] = a_emb[i, :c]
        labels[i, Lc + pl : Lc + pl + c] = a_ids[i, :c]
        if add_eos:
            dec_in[i, Lc + pl + c] = eos_emb
            labels[i, Lc + pl + c] = tok.eos_token_id
            c += 1
        dec_mask[i, : Lc + pl + c] = 1
    return dec_in, dec_mask, labels, None


import ast as _ast

# QA task: supervise the QUERY pathway (bench12 showed replay trained, querying
# never was). Templates deliberately differ from bench12's phrasings; questions
# come from the train split, bench12 from held-out repos — no contamination path.
QA_T = {
    "name": ["What is the name of this function?",
             "State this function's name, exactly.",
             "This function is defined with which name?"],
    "nparams": ["How many parameters does this function take? Exclude self. Reply with just a number.",
                "Count the parameters, not counting self. Number only.",
                "Excluding self, the parameter count is what? Answer with a single number."],
    "params": ["Name every parameter of this function, comma-separated.",
               "Which parameter names does it declare? List them.",
               "Give the parameter list, names only."],
    "call": ["Is `{X}` called anywhere in this function? yes or no.",
             "Does the body invoke `{X}`? Answer yes or no.",
             "yes or no: `{X}` gets called in this function."],
    "lit": ["Quote exactly the {ORD} string literal that appears in this function.",
            "What is the {ORD} string literal in the body, verbatim?",
            "Reproduce the {ORD} quoted string from this function, character for character."],
}
_ORD = ["first", "second", "third", "fourth", "fifth"]


def fn_facts(code):
    try:
        t = _ast.parse(code).body[0]
    except (SyntaxError, IndexError):
        return None
    if not isinstance(t, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
        return None
    doc = _ast.get_docstring(t)
    params = [a.arg for a in t.args.args if a.arg != "self"]
    calls, lits = set(), []
    for node in _ast.walk(t):
        if isinstance(node, _ast.Call):
            f = node.func
            calls.add(f.attr if isinstance(f, _ast.Attribute) else getattr(f, "id", None))
        if isinstance(node, _ast.Constant) and isinstance(node.value, str) \
                and 3 <= len(node.value) <= 60 and node.value != doc:
            lits.append(node.value)
    calls.discard(None)
    return {"name": t.name, "params": params, "n": len(params),
            "calls": sorted(calls), "lits": lits[:5]}


def qa_pair(f, rng, fake_pool):
    """One (question, answer) about facts f. Kind chosen by availability."""
    kinds = ["name", "nparams", "params"] + (["call"] if f["calls"] and fake_pool else []) \
        + (["lit"] if f["lits"] else [])
    kind = rng.choice(kinds)
    t = rng.choice(QA_T[kind])
    if kind == "name":
        return t, f["name"]
    if kind == "nparams":
        return t, str(f["n"])
    if kind == "params":
        return t, (", ".join(f["params"]) if f["params"] else "none")
    if kind == "call":
        if rng.random() < 0.5:
            return t.format(X=rng.choice(f["calls"])), "yes"
        fakes = [c for c in fake_pool if c not in f["calls"] and c != f["name"]]
        if not fakes:
            return t.format(X=rng.choice(f["calls"])), "yes"
        return t.format(X=rng.choice(fakes)), "no"
    k = rng.randrange(len(f["lits"]))
    return t.format(ORD=_ORD[k]), f["lits"][k]


def make_batch_qa(rows, variant_fn, content_rng):
    """[context][\\nQ: question\\nA:][ answer][EOS] — loss on answer(+EOS) only.
    Context variant per batch: vectors (0.6) / raw text (0.2, the answer-format
    rehearsal) / labeled stack with the question naming its target (0.2).
    variant_fn/content_rng: D5 split — the variant coin changes which params
    see grads (raw text skips comp) so it must be the SHARED draw under
    --distributed; question/template content stays rank-local. Legacy passes
    qa_rng.random/qa_rng: consumption order is bit-identical to the old
    inline draws."""
    facts = [_cfacts(c) for c in rows["code"]]
    keep = [i for i, f in enumerate(facts) if f is not None]
    _fb = not keep
    if FSDP_SYNC:
        # FSDP F4 (additional call site): this fallback is a RANK-LOCAL
        # decision (facts parse per rank-local rows) that swaps a qa build for a
        # recon build — under the shared raw-text variant (<0.2, ZERO comp()
        # calls) that is a divergent comp() count. Make it a shared decision:
        # if any rank must fall back, every rank does.
        _all_fb = _sync_any(_fb)
        if _all_fb and not _fb:
            FSDP_STATS["qa_fallback_sync"] += 1  # this rank dragged along
        _fb = _all_fb
    if _fb:
        return make_batch(rows, mask_p=args.mask_p)
    codes = [rows["code"][i] for i in keep]
    facts = [facts[i] for i in keep]
    fake_pool = sorted({c for f in facts for c in f["calls"]})
    variant = variant_fn()

    qs, ans = [], []
    for f in facts:
        q, a = qa_pair(f, content_rng, fake_pool)
        qs.append(q)
        ans.append(" " + a)
    a_enc = _ctok_batch(ans)
    a_ids, a_att = a_enc.input_ids.to(device), a_enc.attention_mask.to(device)

    if variant < 0.2:  # raw-text rehearsal: same skill, uncompressed context
        codes, qs = codes[:8], qs[:8]  # activation cap: raw contexts are full-length
        a_ids, a_att = a_ids[:8], a_att[:8]
        prefix = [_ctok_ids(f"{c}\n\nQ: {q}\nA:") for c, q in zip(codes, qs)]
        zero_vecs = torch.zeros(len(codes), 0, dec_hidden, device=device)
        return assemble(zero_vecs, torch.zeros(len(codes), dtype=torch.long, device=device),
                        a_ids, a_att, prefix_ids=prefix, add_eos=args.eos, use_boundary=False)
    if variant < 0.4:  # stacked: question names its target inside a labeled stack
        codes, qs, facts = codes[:8], qs[:8], facts[:8]  # activation cap: shared ctx x B
        a_ids, a_att = a_ids[:8], a_att[:8]
        names = [f["name"] for f in facts]
        ctx, _, _ = build_stack_ctx(codes, names)
        prompts = [_ctok_ids(f"\n\nQ: In the function `{n}`: {q}\nA:")
                   for n, q in zip(names, qs)]
        return _stack_qa_assemble(ctx, prompts, a_ids, a_att, add_eos=args.eos)
    # default: single-function vectors
    enc = _ctok_batch(codes)
    enc_ids, enc_att = to_bucket(enc.input_ids.to(device), enc.attention_mask.to(device))
    vecs = comp(enc_ids, enc_att)
    n_grp = torch.ceil(enc_att.sum(dim=1).float() / comp.pooling_factor).long()
    prefix = [_ctok_ids(f"\n\nQ: {q}\nA:") for q in qs]
    return assemble(vecs, n_grp, a_ids, a_att, prefix_ids=prefix, add_eos=args.eos)


# --- the final training run pregenerated QA arms (qapre / retr / span) ---------------------------
# All three touch comp (encoder+projector+pooler+boundary) AND the decoder
# trunk/lm_head, exactly like recon — so under --distributed they are
# param-coverage-equivalent to the existing vector arms and find_unused covers
# any residual asymmetry. Record picks are pure-fn rank-keyed draws (D5):
# nothing here is stateful, nothing needs checkpointing.


def _qa_prompt_ids(r):
    """question (+ lettered options iff MCQ) -> no-loss prompt token ids"""
    q = r["question"]
    if r.get("options"):
        q += "".join(f"\n{chr(65 + i)}) {o}" for i, o in enumerate(r["options"]))
    return _ctok_ids(f"\nQ: {q}\nA:")


def _qa_note(recs, span_arm, a_att_cpu):
    """telemetry side-channel: held-out leak tripwire, per-source counts,
    span answer-length histogram; sets CUR_QA_SRC for the loss-EMA hook."""
    global CUR_QA_SRC
    lens = a_att_cpu.sum(dim=1).tolist()
    srcs = []
    for r, ln in zip(recs, lens):
        if r["held_out"]:  # structurally impossible (pools exclude them) — count loudly
            QA_STATS["leak"] += 1
        s_ = r.get("source", "?")
        QA_STATS["src_n"][s_] = QA_STATS["src_n"].get(s_, 0) + 1
        srcs.append(s_)
        if span_arm:
            QA_STATS["span_len"][ln] = QA_STATS["span_len"].get(ln, 0) + 1
    CUR_QA_SRC = srcs


def _qa_pick_solo(pool, rng_):
    """args.batch independent record draws; --token-budget trims the tail so
    sum of context n_tokens fits (min 1 record)."""
    recs = [pool[rng_.randrange(len(pool))] for _ in range(args.batch)]
    if args.token_budget:
        kept, tot = [], 0
        for r in recs:
            t = QA["ntok"][r["row_idx"]]
            if kept and tot + t > args.token_budget:
                break
            kept.append(r)
            tot += t
        recs = kept
    return recs


def _qa_pick_stack(groups, rng_):
    """one stack group (records sharing stack_row_idxs), up to args.batch
    records over the shared context; --token-budget caps the sample count
    (each sample repays the full context) but cannot shrink the stack itself."""
    keys, by = groups
    grp = by[keys[rng_.randrange(len(keys))]]
    recs = grp if len(grp) <= args.batch else rng_.sample(grp, args.batch)
    if args.token_budget:
        stoks = sum(QA["ntok"][i] for i in recs[0]["context"]["stack_row_idxs"])
        recs = recs[: max(1, args.token_budget // max(stoks, 1))]
    return recs


def _draw_qapre(rng_):
    """qapre sub-variant (solo vs stack context) proportional to pool sizes.
    Rank-LOCAL legal: both variants have identical param coverage, so the
    choice never changes which params get grads (unlike the legacy qa raw-text
    coin, which must be shared). --qapre-conc-frac carves a fixed conceptual
    stratum first (short-circuit keeps frac=0 stream-identical to legacy)."""
    if args.qapre_conc_frac > 0 and QA["qapre_conc"] \
            and rng_.random() < args.qapre_conc_frac:
        # realized conceptual-stratum draws (rank-local, like all QA_STATS):
        # the specified 3% is an EXPECTED Bernoulli share — this counter
        # makes the realized share auditable from telemetry (validation)
        QA_STATS["conc_n"] = QA_STATS.get("conc_n", 0) + 1
        return _qa_pick_solo(QA["qapre_conc"], rng_), False
    n_solo, n_stack = len(QA["qapre_solo"]), QA["qapre_stack_n"]
    if n_stack and (not n_solo or rng_.random() < n_stack / (n_solo + n_stack)):
        return _qa_pick_stack(QA["qapre_stack"], rng_), True
    return _qa_pick_solo(QA["qapre_solo"], rng_), False


def make_batch_qa_solo(recs, span_arm=False):
    """qapre-solo / span: [vectors][\\nQ: question\\nA:][ answer][EOS], loss on
    answer+EOS only. Context = the record's corpus row (unshuffled row_idx)."""
    rows = _craw([r["row_idx"] for r in recs])
    enc = _ctok_batch(rows["code"])
    enc_ids, enc_att = to_bucket(enc.input_ids.to(device), enc.attention_mask.to(device))
    vecs = comp(enc_ids, enc_att)
    n_grp = torch.ceil(enc_att.sum(dim=1).float() / comp.pooling_factor).long()
    prefix = [_qa_prompt_ids(r) for r in recs]
    a_tok = _ctok_batch([" " + r["answer"] for r in recs])
    a_ids, a_att = a_tok.input_ids.to(device), a_tok.attention_mask.to(device)
    _qa_note(recs, span_arm, a_tok.attention_mask)
    return assemble(vecs, n_grp, a_ids, a_att, prefix_ids=prefix, add_eos=True)


def make_batch_qa_stackarm(recs):
    """qapre-stack / retr: shared labeled stack context from the record's
    stack_row_idxs (same assembly as make_batch_stack's context), question
    names or describes the target per the record; loss on answer+EOS."""
    rows = _craw(list(recs[0]["context"]["stack_row_idxs"]))
    # short names everywhere (fixed recipe stack-header decision, the fixed recipe):
    # corpus func_name is 56.5% dotted Class.method, but retrieval ANSWERS are
    # short by construction and the legacy stack arm labels short — one
    # convention across arms
    names = (["" if n is None else str(n).split(".")[-1] for n in rows["func_name"]]
             if "func_name" in rows else
             [(m.group(1) if (m := _DEF.search(c)) else f"fn{i}")
              for i, c in enumerate(rows["code"])])
    ctx, _, _ = build_stack_ctx(rows["code"], names)
    prompts = [_qa_prompt_ids(r) for r in recs]
    a_tok = _ctok_batch([" " + r["answer"] for r in recs])
    a_ids, a_att = a_tok.input_ids.to(device), a_tok.attention_mask.to(device)
    _qa_note(recs, False, a_tok.attention_mask)
    global CUR_WMAP
    CUR_WMAP = None  # pregen arms are unweighted (surprisal keys the corpus rows, not answers)
    return _stack_qa_assemble(ctx, prompts, a_ids, a_att, add_eos=True)


DISTRACT_LEN = 128  # fixed length: one extra shape family, not one per batch


def with_distractor(batch, dist_rng):
    """Prepend DISTRACT_LEN tokens of unrelated code as plain text before every
    sample: the vectors must keep working mid-window, not only at position 0.

    NOTE (validation): when the drawn distractor is SHORTER than 128
    tokens, the EOS filler extending it to the fixed length is attention-
    VISIBLE in the legacy path (mask all-ones below) — filler EOS runs are
    real context the model reads. That is the distribution earlier configurations trained
    on; --mask-distractor-pad (default OFF) masks the filler instead. The
    default must only change by an explicit recipe change (fixed recipe) because flipping it changes the realized training distribution
    vs the whole lineage."""
    dec_in, dec_mask, labels, pmap = batch
    ids = _ctok_ids(train[dist_rng.randrange(len(train))]["code"])[:DISTRACT_LEN]
    n_real = len(ids)
    ids += [tok.pad_token_id] * (DISTRACT_LEN - len(ids))
    B = dec_in.shape[0]
    demb = embed_table(torch.tensor([ids], device=device)).expand(B, -1, -1).to(dec_in.dtype)
    dec_in = torch.cat([demb, dec_in], dim=1)
    dmask = torch.ones(B, DISTRACT_LEN, dtype=torch.long, device=device)
    if args.mask_distractor_pad and n_real < DISTRACT_LEN:
        dmask[:, n_real:] = 0  # filler EOS: masked, not context
    dec_mask = torch.cat([dmask, dec_mask], dim=1)
    labels = torch.cat([torch.full((B, DISTRACT_LEN), -100, dtype=torch.long, device=device), labels], dim=1)
    if pmap is not None:
        pmap = torch.cat([torch.zeros(B, DISTRACT_LEN, dtype=torch.bool, device=device), pmap], dim=1)
    global CUR_WMAP
    if CUR_WMAP is not None:  # keep the surprisal channel aligned with the shift
        CUR_WMAP = torch.cat(
            [torch.ones(B, DISTRACT_LEN, dtype=torch.float32, device=device), CUR_WMAP], dim=1)
    return dec_in, dec_mask, labels, pmap


MIX = {}
for part in args.mix.split(","):
    k, v = part.split("=")
    MIX[k.strip()] = float(v)
assert abs(sum(MIX.values()) - 1.0) < 1e-6, f"mix weights must sum to 1: {MIX}"
_QA_ARMS = ("qapre", "retr", "span")  # the final training run pregenerated arms (--qa-data)
assert set(MIX) <= {"recon", "cont", "desc", "plain", "stack", "qa", *_QA_ARMS}, \
    f"unknown task in {MIX}"
if set(_QA_ARMS) & set(MIX):
    assert QA is not None, f"--mix uses {set(_QA_ARMS) & set(MIX)} but --qa-data is not set"
    _pool_n = {"qapre": len(QA["qapre_solo"]) + QA["qapre_stack_n"],
               "retr": sum(len(v) for v in QA["retr"][1].values()),
               "span": len(QA["span"])}
    for _k in set(_QA_ARMS) & set(MIX):
        assert _pool_n[_k] > 0, f"mix arm {_k}: --qa-data pool is empty"

PF_MIX = None
if args.pf_mix:
    PF_MIX = {}
    for part in args.pf_mix.split(","):
        k, v = part.split("=")
        PF_MIX[int(k)] = float(v)
    assert abs(sum(PF_MIX.values()) - 1.0) < 1e-6, f"pf-mix weights must sum to 1: {PF_MIX}"
GEN_PFS = sorted(PF_MIX) if PF_MIX else [args.pf]


def loss_on(dec_in, dec_mask, labels, weighted=False, pmap=None):
    """Per-token CE, optionally rarity-weighted. Unweighted == HF's .loss, so
    eval metrics stay comparable across all runs regardless of training flags.

    pmap (training only, with --prose-weight): prose positions get a flat
    args.prose_weight instead of the rarity-boosted weight.

    --chunked-loss grades only label positions, in checkpointed slices, so the
    BxTxV logits tensor is never materialized (it, not the model, caps batch)."""
    global CUR_WMAP, CUR_PSHORT, CUR_ROWIDS
    wmap_full, CUR_WMAP = CUR_WMAP, None  # consume-and-clear the batch channel
    pshort, CUR_PSHORT = CUR_PSHORT, None
    rowids, CUR_ROWIDS = CUR_ROWIDS, None
    # (validation): margin telemetry is TRAINING-population only. The
    # pre-validation accumulators also counted every no_grad eval loss_on call
    # (floor/ceiling/apf probes), so the unconditional floor/ceiling dedup
    # silently changed the margin population at the next %50 line — the
    # "observationally exact" claim was false for this one channel. Gating on
    # grad mode makes the dedup margin-invariant (harness an earlier configuration); deviation
    # from the earlier configurations stdout-margin population (which included eval calls)
    # is noted in fixed recipe Kill criteria never read margin lines.
    _margin_live = torch.is_grad_enabled()
    if (weighted and args.self_prefix_frac > 0
            and dec_in.shape[1] <= args.self_prefix_maxlen
            and torch.rand(()).item() < args.self_prefix_frac):
        # exposure: one no-grad forward finds where the model would commit
        # WRONG right now; those input positions carry its token instead of
        # gold (labels stay gold) — commitment errors become training signal.
        with torch.no_grad():
            # sub-batched: the full-batch no-grad forward's transient peak OOM'd
            # a 95GB card by 112MiB on top of the recipe's 89.7GB (validation)
            B, L, _ = dec_in.shape
            prev_h_parts = []
            for b0 in range(0, B, 4):
                h0 = dec_trunk(inputs_embeds=dec_in[b0 : b0 + 4],
                               attention_mask=dec_mask[b0 : b0 + 4]).last_hidden_state
                prev_h_parts.append(h0[:, :-1])
            prev_h = torch.cat(prev_h_parts, dim=0).flatten(0, 1)
            del prev_h_parts
            lb0 = labels[:, 1:].flatten()
            idx = (lb0 != -100).nonzero(as_tuple=True)[0]
            preds = torch.empty(idx.shape[0], dtype=torch.long, device=dec_in.device)
            for i in range(0, idx.shape[0], args.loss_chunk):
                j = idx[i : i + args.loss_chunk]
                preds[i : i + args.loss_chunk] = lm_head(prev_h[j]).argmax(-1)
            del prev_h
            wrongly = preds != lb0[idx]
            coin = torch.rand(idx.shape[0], device=dec_in.device) < args.self_prefix_p
            take = wrongly & coin
            if take.any():
                b = idx[take] // (L - 1)
                t = idx[take] % (L - 1) + 1
                dec_in = dec_in.clone()
                dec_in[b, t] = embed_table(preds[take]).to(dec_in.dtype)
    hidden = dec_trunk(inputs_embeds=dec_in, attention_mask=dec_mask).last_hidden_state
    lb = labels[:, 1:].flatten()
    pm = pmap[:, 1:].flatten() if (weighted and pmap is not None) else None
    psh = None
    if (pm is not None and pshort is not None and args.prose_weight_short is not None
            and pshort.shape[0] == labels.shape[0]):
        psh = pshort[:, None].expand(-1, labels.shape[1])[:, 1:].flatten()
    wm = (wmap_full[:, 1:].flatten()
          if (weighted and args.surprisal_weight > 0 and wmap_full is not None) else None)
    if args.chunked_loss:
        h = hidden[:, :-1].flatten(0, 1)
        keep = lb != -100
        h, lb = h[keep], lb[keep]
        if pm is not None:
            pm = pm[keep]
        if wm is not None:
            wm = wm[keep]
        if psh is not None:
            psh = psh[keep]
        num = torch.zeros((), dtype=torch.float32, device=h.device)
        den = torch.zeros((), dtype=torch.float32, device=h.device)
        # active-norm hinge accumulates OUTSIDE num/den so its normalizer is
        # the active-tie count, not the CE token mass
        rknum = torch.zeros((), dtype=torch.float32, device=h.device)
        rkact = 0
        bcnt = bix = None
        if rowids is not None and RANK_K > 0 and pm is not None and RANK_GATE:
            B, L = labels.shape
            bix = torch.arange(B, device=h.device).repeat_interleave(L - 1)[keep]
            bcnt = torch.zeros(B, dtype=torch.long, device=h.device)
        for i in range(0, h.shape[0], args.loss_chunk):
            hc, lbc = h[i : i + args.loss_chunk], lb[i : i + args.loss_chunk]
            ce, margin = (_ckpt(_ce_slice, hc, lbc, use_reentrant=False)
                          if h.requires_grad else _ce_slice(hc, lbc))
            # telemetry OUTSIDE _ce_slice: the checkpointed fn re-runs at
            # backward and would double-count a global accumulator.
            # _margin_live: training calls only () — eval loss_on calls
            # no longer pollute the margin population.
            with torch.no_grad():
                if not _margin_live:
                    pass
                elif TELEM_BATCH:
                    # device-side: three tiny kernels, zero host syncs; drained
                    # at the %50 boundary (--telemetry-batched)
                    MARGIN_ACC_F.add_(margin.sum())
                    MARGIN_ACC_I[0] += margin.numel()
                    MARGIN_ACC_I[1] += (margin < 0).sum()
                    MARGIN_ACC_I[2] += (margin < args.margin_tau).sum()
                else:
                    MARGIN_STATS[0] += float(margin.sum())
                    MARGIN_STATS[1] += margin.numel()
                    MARGIN_STATS[2] += int((margin < 0).sum())
                    MARGIN_STATS[3] += int((margin < args.margin_tau).sum())
            if RANK_K > 0 and pm is not None and RANK_GATE:
                # the ranking-hinge design: hinge on the gap itself, non-prose tokens only.
                # Magnitude auto-prioritizes corrections (negative gap = big
                # hinge) over consolidation (thin-but-right = small hinge).
                npm = ~pm[i : i + args.loss_chunk]
                hin = torch.relu(args.rank_margin - margin) * npm.float()
                if args.rank_active_norm:
                    rknum = rknum + hin.sum()
                else:
                    num = num + RANK_K * hin.sum()
                with torch.no_grad():
                    act = (margin < args.rank_margin) & npm
                    conf = (margin >= args.rank_margin) & npm
                    n_act = int(act.sum())
                    rkact += n_act
                    RANK_STATS[0] += int(npm.sum())
                    RANK_STATS[1] += n_act
                    RANK_STATS[2] += float(hin.sum())
                    RANK_STATS[3] += float((margin * npm).sum())
                    RANK_STATS[4] += float((margin * conf).sum())
                    RANK_STATS[5] += int(conf.sum())
                    if bcnt is not None and n_act:
                        ai = bix[i : i + args.loss_chunk][act]
                        bcnt.scatter_add_(0, ai, torch.ones_like(ai))
            w = torch.ones_like(ce)
            if weighted and args.rare_weight > 0:
                w = w + args.rare_weight * rarity_w[lbc]
            if weighted and MARGIN_K > 0:
                w = w * (1.0 + MARGIN_K * torch.sigmoid(-margin.detach() / args.margin_tau))
            if wm is not None:
                w = w * wm[i : i + args.loss_chunk]
            if pm is not None:
                pw = torch.full_like(w, args.prose_weight)
                if psh is not None:  # short docstrings get the verbatim floor
                    pw = torch.where(psh[i : i + args.loss_chunk],
                                     torch.full_like(w, args.prose_weight_short), pw)
                w = torch.where(pm[i : i + args.loss_chunk], pw, w)
            num = num + (ce * w).sum()
            den = den + w.sum()
        if bcnt is not None:
            with torch.no_grad():  # one sync per recon step, opt-in (--hard-mine)
                cl = bcnt.tolist()
            HARD_STATS[1] += 1
            HARD_STATS[2] += float(sum(cl))
            for rid, c in zip(rowids, cl):
                if c or rid in HARD_EMA:  # zero-count unseen rows never enter
                    e = 0.8 * HARD_EMA.get(rid, float(c)) + 0.2 * c
                    if e < 0.05:  # decayed to noise: self-prune
                        HARD_EMA.pop(rid, None)
                    else:
                        HARD_EMA[rid] = e
        out = num / den.clamp_min(1)
        if args.rank_active_norm and rkact > 0:
            out = out + RANK_K * rknum / rkact
        return out
    lg = lm_head(hidden[:, :-1]).flatten(0, 1).float()
    ce = torch.nn.functional.cross_entropy(lg, lb, ignore_index=-100, reduction="none")
    w = (lb != -100).float()
    if weighted and args.rare_weight > 0:
        w = w * (1.0 + args.rare_weight * rarity_w[lb.clamp_min(0)])
    if weighted and MARGIN_K > 0:
        with torch.no_grad():
            safe_lb = lb.clamp_min(0)
            top2 = lg.detach().topk(2, dim=-1)
            true_lg = lg.detach().gather(-1, safe_lb.unsqueeze(-1)).squeeze(-1)
            rival = torch.where(top2.indices[:, 0] == safe_lb,
                                top2.values[:, 1], top2.values[:, 0])
            mw = 1.0 + MARGIN_K * torch.sigmoid(-(true_lg - rival) / args.margin_tau)
        w = w * torch.where(lb != -100, mw, torch.ones_like(mw))
    if wm is not None:
        w = w * wm
    if pm is not None:
        pw = torch.full_like(w, args.prose_weight)
        if psh is not None:
            pw = torch.where(psh, torch.full_like(w, args.prose_weight_short), pw)
        w = torch.where(pm & (lb != -100), pw, w)
    return (ce * w).sum() / w.sum().clamp_min(1)


MICRO_TC = [0.0]  # build->compute wall boundary inside micro_loss (timing channel)
# --token-budget realized-batch telemetry (log-everything rule): [rows, real
# (unpadded) decoder tokens, micro-batches] since the last steps.jsonl write —
# the honest record of what "effective batch" actually was under the budget
TB_STATS = [0, 0, 0]
if TELEM_BATCH:
    TB_ACC = torch.zeros(3, dtype=torch.int64, device=device)  # rows, tok, n


def micro_loss(task, rows, row_ids, step, mc):
    """The ONE micro-batch body: arm build + loss (+ scaffold). Legacy calls it
    directly; --distributed calls it through DDPShell.forward so the reducer
    sees exactly one forward per micro-step (inner comp() multiplicity — stack
    chunks, distractor — is invisible at the shell boundary, D1/D3). mc carries
    the RNG streams: legacy passes the module-global streams so consumption
    order stays bit-identical to the pre-refactor inline body."""
    global CUR_ROWIDS, CUR_SCAF_LOSS, RANK_GATE, CUR_QA_SRC
    with FT_AC():  # full-ft fp32 master weights: build+forward compute in bf16
        if task == "cont":
            dec_in, dec_mask, labels, pmap = make_batch_cont(rows)
        elif task == "desc":
            dec_in, dec_mask, labels, pmap = make_batch_desc(rows)
        elif task == "plain":
            dec_in, dec_mask, labels, pmap = make_batch_plain(rows)
        elif task == "stack":
            dec_in, dec_mask, labels, pmap = make_batch_stack(rows, mc["stack_rng"])
        elif task == "qa":
            dec_in, dec_mask, labels, pmap = make_batch_qa(rows, mc["qa_variant"], mc["qa_rng"])
        elif task == "qapre":
            _recs, _st = _draw_qapre(mc["qapre_rng"])
            dec_in, dec_mask, labels, pmap = (make_batch_qa_stackarm(_recs) if _st
                                              else make_batch_qa_solo(_recs))
        elif task == "retr":
            dec_in, dec_mask, labels, pmap = make_batch_qa_stackarm(
                _qa_pick_stack(QA["retr"], mc["retr_rng"]))
        elif task == "span":
            dec_in, dec_mask, labels, pmap = make_batch_qa_solo(
                _qa_pick_solo(QA["span"], mc["span_rng"]), span_arm=True)
        else:
            if args.hard_mine > 0:
                k = int(args.batch * args.hard_mine)
                if k > 0 and HARD_EMA:
                    pool = sorted(HARD_EMA.items(), key=lambda kv: -kv[1])[:2048]
                    picks = mc["hm_rng"].choices([p[0] for p in pool],
                                                 weights=[p[1] for p in pool], k=k)
                    row_ids = picks + row_ids[k:]
                    rows = train[row_ids]
                    HARD_STATS[0] += k
                CUR_ROWIDS = row_ids  # attribution channel (recon only)
            batch = make_batch(rows, mask_p=args.mask_p)
            if args.distractor_frac > 0 and mc["dist_coin"]() < args.distractor_frac:
                batch = with_distractor(batch, mc["dist_rng"])
            dec_in, dec_mask, labels, pmap = batch
        _rebuild = bool((labels != -100).sum() == 0)  # e.g. a desc batch with no docstrings
        if FSDP_SYNC:
            # FSDP F4 (second additional call site): the rebuild adds
            # one comp() forward on THIS rank only. Shared decision (any rank
            # -> all ranks rebuild; the dropped first build is symmetric across
            # ranks, so backward collectives still match). Unconditional
            # per-micro collective: ~8 bytes, and doubles as a liveness point.
            # Validation of --telemetry-batched: this allreduce CANNOT be
            # batched/deferred — its result gates whether THIS micro's batch is
            # rebuilt before loss_on, i.e. it is consumed synchronously for
            # correctness (an async decision would need speculative builds of
            # both branches and would still have to join before the forward).
            # Only the rebuild_sync counter below is observational, and it is
            # already sync-free Python. Left per-micro by design.
            _all_rb = _sync_any(_rebuild)
            if _all_rb and not _rebuild:
                FSDP_STATS["rebuild_sync"] += 1  # this rank dragged along
            _rebuild = _all_rb
        if _rebuild:
            CUR_QA_SRC = None  # rebuilt as recon: don't attribute its loss to qa sources
            dec_in, dec_mask, labels, pmap = make_batch(rows, mask_p=args.mask_p)
        if args.token_budget:  # realized-batch telemetry (flag-gated)
            if TELEM_BATCH:
                TB_ACC[0] += dec_in.shape[0]
                TB_ACC[1] += dec_mask.sum().to(torch.int64)  # device op, no sync
                TB_ACC[2] += 1
            else:
                TB_STATS[0] += dec_in.shape[0]
                TB_STATS[1] += int(dec_mask.sum())  # one host sync per micro
                TB_STATS[2] += 1
        MICRO_TC[0] = time.time()
        RANK_GATE = task == "recon" if args.rank_recon_only else True
        loss = loss_on(dec_in, dec_mask, labels, weighted=(task != "plain"), pmap=pmap)
    if CUR_QA_SRC is not None:
        # per-source loss EMA for the pregen arms (rank-local; one .item()/qa micro)
        _li = float(loss.detach())
        for _s in CUR_QA_SRC:
            QA_SRC_EMA[_s] = 0.98 * QA_SRC_EMA.get(_s, _li) + 0.02 * _li
        CUR_QA_SRC = None
    if CUR_SCAF_LOSS is not None:
        # cosine anneal lambda0 -> 0 over scaffold-steps; after that the AST
        # scaffold is mathematically absent and the model is fully free
        _t = min(step / max(args.scaffold_steps, 1), 1.0)
        _lam = args.span_scaffold * 0.5 * (1.0 + math.cos(math.pi * _t))
        if _lam > 0:
            loss = loss + _lam * CUR_SCAF_LOSS
        CUR_SCAF_LOSS = None
    return loss


class DDPShell(torch.nn.Module):
    """One container module (D1) so DDP's reducer spans comp + trunk + lm_head
    with a single bucket schedule and one no_sync; forward IS the whole
    micro-batch computation. Registration only — micro_loss keeps using the
    module globals, which reference the same objects held here."""

    def __init__(self, comp, dec_trunk, lm_head):
        super().__init__()
        self.comp, self.dec_trunk, self.lm_head = comp, dec_trunk, lm_head

    def forward(self, task, rows, row_ids, step, mc):
        return micro_loss(task, rows, row_ids, step, mc)


@torch.no_grad()
def evaluate(pf=None, with_floor=True):
    """held-out recon loss: with vectors / no context (floor) / text copy (ceiling).
    Default pf = base --pf (the comparable column); pass pf to probe another
    ratio (the in-loop TF-capacity tripwire — an earlier configuration's 8x crack was only found by
    ad-hoc mid-run scripts; now it's a logged column).

    with_floor=False (floor/ceiling dedup, unconditional — exact math): the
    floor (no-ctx) and ceiling (copy) losses do not depend on pf, so the
    per-apf probe calls skip recomputing them and reuse the base call's values
    at the call site. The skipped forwards consume no RNG (mask_p=0, weighted
    off), so the vec loss is byte-identical to the pre-dedup path (rehearsal
    an earlier configuration; env EVAL_NO_DEDUP=1 re-runs the old path for the A/B).
    Forwards run under EVAL_AC (--eval-autocast; nullcontext by default)."""
    _pf = comp.pooling_factor
    comp.pooling_factor = pf or args.pf
    comp.eval()
    if args.full_ft:
        decoder.eval()
    tot_v, tot_n, tot_c, n = 0.0, 0.0, 0.0, 0
    with EVAL_AC():
        for s in range(0, len(val_fixed), args.batch):
            rows = val_fixed[s : s + args.batch]
            dec_in, dec_mask, labels, _ = make_batch(rows)
            tot_v += loss_on(dec_in, dec_mask, labels).item()

            if with_floor:
                enc = tok(rows["code"], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(device)
                ce = embed_table(enc.input_ids)
                lab = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
                # floor: predict the code given nothing
                tot_n += loss_on(ce, enc.attention_mask, lab).item()
                # ceiling: predict the code given the code itself as raw text ([code][code])
                copy_in = torch.cat([ce, ce], dim=1)
                copy_mask = torch.cat([enc.attention_mask, enc.attention_mask], dim=1)
                copy_lab = torch.cat([torch.full_like(lab, -100), lab], dim=1)
                tot_c += loss_on(copy_in, copy_mask, copy_lab).item()
            n += 1
    comp.train()
    if args.full_ft:
        decoder.train()
    comp.pooling_factor = _pf
    if device == "mps":
        torch.mps.empty_cache()
    if not with_floor:
        return tot_v / n, None, None
    return tot_v / n, tot_n / n, tot_c / n


# --- generation eval: the honest metric, in the loop --------------------------
import difflib

from compressor.anchors import build_anchored_context
from compressor.genutil import batched_generate

canaries = [json.loads(l) for l in open(ROOT / "data" / "canaries.jsonl")][: args.gen_canaries]

# the ranking-hinge design placement probe: per-class margin stats on the fixed v2 canaries,
# TF-aligned exactly like exp-26 (same tokenization, no EOS/pad) so every
# reading is comparable to the $0 sims that locked the design
# (the token-class probe). Classes are telemetry ONLY — never weighted.
PROBE_META = None
if args.rank_loss > 0:
    from _tokclass import token_classes as _token_classes
    PROBE_META = []
    for _c in [json.loads(l) for l in open(ROOT / "data" / "canaries_v2.jsonl")]:
        _cl, _ids, _ = _token_classes(tok, _c["code"])
        PROBE_META.append((_c["func_name"], _c["code"], _cl, _ids))


@torch.no_grad()
def rank_probe(pf=None):
    """Per-class {n, act(<m*), wrong(<0), mean margin} at TF on v2 canaries.
    Caller is expected to have EMA weights swapped in (gen-eval block)."""
    _pf = comp.pooling_factor
    comp.pooling_factor = pf or args.pf
    comp.eval()
    if args.full_ft:
        decoder.eval()
    agg = {}
    with EVAL_AC():  # --eval-autocast (nullcontext by default)
        for _name, code, cls_l, ids in PROBE_META:
            vecs = comp.compress(code, device).to(embed_table.weight.dtype)
            seq = torch.cat(
                [vecs, embed_table(torch.tensor(ids, device=device).view(1, -1))], dim=1)
            hid = dec_trunk(inputs_embeds=seq,
                            attention_mask=torch.ones(seq.shape[:2], dtype=torch.long,
                                                      device=device)).last_hidden_state
            P = vecs.shape[1]
            # keep decoder dtype — lm_head is bf16 on CUDA; _ce_slice floats the
            # logits itself (fp32 h crashed the box preflight, validation)
            h = hid[0, P - 1 : P - 1 + len(ids)]
            for j in range(0, h.shape[0], args.loss_chunk):
                lbc = torch.tensor(ids[j : j + args.loss_chunk], device=device)
                _, m = _ce_slice(h[j : j + args.loss_chunk], lbc)
                for k, mm in zip(cls_l[j : j + args.loss_chunk], m.tolist()):
                    a = agg.setdefault(k, [0, 0, 0, 0.0])
                    a[0] += 1
                    a[1] += mm < args.rank_margin
                    a[2] += mm < 0
                    a[3] += mm
    comp.train()
    if args.full_ft:
        decoder.train()
    comp.pooling_factor = _pf
    return {k: {"n": a[0], "act": a[1], "wrong": a[2], "m": round(a[3] / a[0], 2)}
            for k, a in sorted(agg.items())}


@torch.no_grad()
def drift_probe(n=8):
    """drift curve: on n fixed val functions, how far is the pooler's
    actual per-span attention mass from the AST+surprisal proposal? Logged
    every eval. Drift AND improvement = learning superseded syntax; no drift =
    the AST prior was near-optimal. Either is the experiment's answer."""
    comp.eval()
    if args.full_ft:
        decoder.eval()
    agg = {}
    codes = [val[i]["code"] for i in range(n)]
    enc = tok(codes, return_tensors="pt", padding=True, add_special_tokens=False,
              return_offsets_mapping=True)
    enc_ids, enc_att = to_bucket(enc.input_ids.to(device), enc.attention_mask.to(device))
    comp(enc_ids, enc_att)
    pa = comp.projector.latent_pooler.last_attn          # (B,H,G,T)
    mass = pa.mean(1).sum(1) * enc_att                   # (B,T)
    mass = (mass / mass.sum(1, keepdim=True).clamp_min(1e-8)).cpu()
    tgt, _ = scaffold_target(codes, enc.offset_mapping.tolist(), enc_ids.shape[1])
    for i, code in enumerate(codes):
        offs = enc.offset_mapping[i].tolist()
        for kind, s, e in _seg_spans(code):
            tk = [t for t, (cs, ce) in enumerate(offs) if ce > cs and s <= cs < e]
            if not tk:
                continue
            a = agg.setdefault(kind, [0.0, 0.0, 0])
            a[0] += float(mass[i, tk].sum())
            a[1] += float(tgt[i, tk].sum())
            a[2] += 1
    comp.train()
    if args.full_ft:
        decoder.train()
    return {k: {"mass": round(m / c, 4), "prop": round(p / c, 4)}
            for k, (m, p, c) in agg.items()}


@torch.no_grad()
def gen_eval(dump_step=None):
    """Free-running canary reconstruction: (exact, code_exact, avg_similarity).
    code_exact ignores docstring/comment wording (the code-focused metric).
    Runs at the CURRENT comp.pooling_factor (call sites loop GEN_PFS).

    Canaries generate in batches of --gen-batch via batched_generate (the
    serial batch-1 token loop was the wall-clock hog on weak-CPU hosts —
    validation Swiss box: suite 5x slower, GPU idle)."""
    comp.eval()
    if args.full_ft:
        decoder.eval()
    exact, code_exact, sims = 0, 0, []
    with EVAL_AC():  # --eval-autocast (nullcontext by default)
        if args.anchor_frac > 0:
            vl = [build_anchored_context(comp, tok, embed_table, c["code"], device,
                                         args.anchor_frac, rarity_w)[0] for c in canaries]
        else:
            vl = [comp.compress(c["code"], device)[0] for c in canaries]  # boundary-wrapped
        outs = batched_generate(decoder, vl, [c["n_tokens"] + 32 for c in canaries],
                                batch_size=args.gen_batch)
    for c, out in zip(canaries, outs):
        gen = tok.decode(out, skip_special_tokens=True)
        sims.append(difflib.SequenceMatcher(None, c["code"], gen[: len(c["code"]) + 200]).ratio())
        exact += gen.strip().startswith(c["code"].strip())
        code_exact += strip_prose_text(gen).startswith(strip_prose_text(c["code"]))
        if dump_step is not None and IS_MAIN:
            # full mid-run generations on the record — every prior run needed
            # ad-hoc ssh scripts to SEE what the model was actually writing.
            # IS_MAIN: under FSDP every rank runs gen_eval (F10) — write once.
            with open(run_dir / "gen_samples.jsonl", "a") as gs:
                gs.write(json.dumps({"step": dump_step, "pf": comp.pooling_factor,
                                     "name": c["func_name"], "sim": round(sims[-1], 4),
                                     "gen": gen[: len(c["code"]) + 400]}) + "\n")
    comp.train()
    if args.full_ft:
        decoder.train()
    return exact, code_exact, sum(sims) / len(sims)


# --- equivalence test: chunked loss must equal full loss, in value and grad ----
if args.test_equivalence:
    rows = val_fixed[:4]

    def measure(chunked, weighted):
        args.chunked_loss = chunked
        comp.projector.zero_grad(set_to_none=True)
        torch.manual_seed(0)  # fresh graph per pass, identical inputs
        dec_in, dec_mask, labels, pmap_ = make_batch(rows)
        loss = loss_on(dec_in, dec_mask, labels, weighted=weighted, pmap=pmap_)
        loss.backward()
        return loss.item(), torch.cat([p_.grad.flatten().clone() for p_ in comp.projector.parameters()])

    def rel(a, b):
        return (a - b).norm().item() / max(a.norm().item(), 1e-9)

    # Grad tolerance is precision-bound: splitting the vocab-dim matmul changes
    # accumulation order, and low-precision rounding amplifies through the trunk
    # backward. Verified validation: bf16 1.9e-2 -> fp32 4e-4 (scales with
    # precision => rounding, not math). Loss values match bitwise in both.
    grad_tol = 5e-2 if args.dec_dtype == "bfloat16" else 1e-3
    ok = True
    for weighted in (False, True):
        v_a, g_a = measure(False, weighted)   # full, run A
        _, g_b = measure(False, weighted)     # full, run B -> backward noise floor
        v_c, g_c = measure(True, weighted)    # chunked
        dv = abs(v_a - v_c) / max(abs(v_a), 1e-9)
        noise, dg = rel(g_a, g_b), rel(g_a, g_c)
        line_ok = dv < 1e-4 and dg < max(3 * noise, grad_tol)
        ok &= line_ok
        print(f"{'weighted' if weighted else 'unweighted':>10}: full {v_a:.6f} | chunked {v_c:.6f} | "
              f"rel-dloss {dv:.2e} | grad-noise-floor {noise:.2e} | rel-dgrad {dg:.2e} | "
              f"{'PASS' if line_ok else 'FAIL'}")
    print("EQUIVALENCE " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)

# --- train --------------------------------------------------------------------
import math

# --- FSDP2 wrap (FSDP DESIGN F1/F3/F6): MUST precede param-group construction —
# fully_shard swaps every nn.Parameter for a DTensor IN PLACE, so any param
# list captured before this point is stale (p.grad stays None forever, step()
# becomes a silent no-op — danger #7). The ft_/lora_ lists captured at model
# build are re-derived below. Legacy + DDP paths: this whole block is a no-op.
shell = None
_FSDP_BLOCKS = []  # per-block fsdp modules (eval bracket needs them)
_DT = None         # DTensor class when FSDP is live (isinstance guards)
if FSDP_ON:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor as _DT

    # explicit mesh: the default mesh init auto-detects an accelerator and
    # crashes on MPS hosts (torch.mps lacks is_initialized) — and the gloo
    # rehearsal must pin cpu regardless
    _mesh = init_device_mesh("cuda" if device.startswith("cuda") else "cpu", (WORLD,))
    _mp_kw = {}
    if args.fsdp_param_dtype == "bfloat16":
        # reduce_dtype MUST be explicit fp32: left None it resolves to the
        # bf16 grad dtype and silently reduces grads in bf16 (F5/F12;
        # #186998's failure mode). NOTE the final sharded grad is cast back to
        # the fp32 master dtype AFTER the reduction (_fsdp_collectives.py:
        # _to_dtype_if_needed(reduce_output, orig_dtype), verified 2.12.1),
        # so a bf16 reduce is INVISIBLE to any grad-dtype assert — the in-loop
        # tripwire therefore introspects each group's RESOLVED _reduce_dtype
        # at first backward (re-armed after unfreezes, #186998's trigger).
        if args.fsdp_test_bf16_reduce:
            # TEST ONLY (negative control): the exact trap the comment
            # above forbids; the introspection tripwire must abort the run.
            _mp_kw["mp_policy"] = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        else:
            _mp_kw["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    _shell = DDPShell(comp, dec_trunk, lm_head)
    _rs_blk = args.fsdp_reshard == "full"
    # bottom-up: one group per transformer block; the ROOT group (both
    # embed_tokens + tied lm_head + final norms + whole projector) never
    # reshards after forward — variable-count lm_head/_ce_slice calls must
    # post no collective (F4), and tied lm_head/embed_tokens must never be
    # split across groups (F6). dec_trunk is eager here (compile asserted off).
    # mp_policy goes on the BLOCK groups ONLY (2.7B of 3.37B params). The root
    # group must stay fp32: eval paths call comp()/dec_trunk/lm_head DIRECTLY,
    # outside FT_AC autocast AND outside the shell forward whose pre-hook would
    # cast inputs — bf16-gathered root params there feed fp32 activations into
    # strict-same-dtype kernels (latent-pooler LayerNorm/matmul, projector +
    # lm_head Linear; the pooler is fp32-by-design, model.py '.float()' island)
    # and crash on CUDA and CPU alike (measured: validation before the fix).
    # Blocks are safe everywhere: cast_forward_inputs=True casts each block's
    # hidden-state input to bf16 at its own boundary, and the fp32-weight
    # final RMSNorm promotes block output back to fp32 at both exits.
    for _blk in list(comp.encoder.layers) + list(dec_trunk.layers):
        fully_shard(_blk, mesh=_mesh, reshard_after_forward=_rs_blk, **_mp_kw)
        _FSDP_BLOCKS.append(_blk)
    fully_shard(_shell, mesh=_mesh, reshard_after_forward=False)
    # F10 layer-2 nrf: per-block forward-coverage hooks (— torch 2.13
    # dropped the warning the string filter matched). Python-int increments in
    # a pre-hook: no device work, no syncs; counts only GRAD-ENABLED forwards
    # — every eval path here runs under no_grad (_fsdp_eval_ctx / @no_grad),
    # so grad-enabled == training-path, and eval passes can't mask a dead
    # training path. (NOT module.training: a frozen encoder legitimately
    # rides in eval mode through early steps yet its forwards count.)
    _BLK_FWD_COV.extend([0] * len(_FSDP_BLOCKS))

    def _mk_cov_hook(_i):
        def _cov(module, _args):
            if torch.is_grad_enabled():
                _BLK_FWD_COV[_i] += 1
        return _cov
    for _i, _blk in enumerate(_FSDP_BLOCKS):
        _blk.register_forward_pre_hook(_mk_cov_hook(_i))
    shell = _shell
    # re-capture every pre-wrap param list (stale-param danger #7)
    if args.full_ft:
        ft_enc_ps = [q for q in comp.encoder.parameters() if q.requires_grad]
        ft_dec_ps = [q for q in decoder.parameters() if q.requires_grad]
    if lora_ps:
        lora_ps = lora_parameters(decoder)
    if enc_lora_ps:
        enc_lora_ps = lora_parameters(comp.encoder)
    _n_full = sum(q.numel() for q in _shell.parameters())
    _n_loc = sum(q._local_tensor.numel() if isinstance(q, _DT) else q.numel()
                 for q in _shell.parameters())
    print(f"FSDP2: world {WORLD} | batch {args.batch}/rank x accum {args.grad_accum} "
          f"= eff {WORLD * args.batch * args.grad_accum} | backend {dist.get_backend()} | "
          f"reshard {args.fsdp_reshard} | param_dtype {args.fsdp_param_dtype}"
          f"{' (blocks; root fp32)' if args.fsdp_param_dtype == 'bfloat16' else ''} "
          f"reduce_dtype float32 | groups {len(_FSDP_BLOCKS)}+root | "
          f"params {_n_full / 1e6:.1f}M full -> {_n_loc / 1e6:.1f}M/rank sharded | "
          f"count-sync {'OFF (TEST ONLY)' if args.fsdp_no_count_sync else 'on'} | "
          f"accum-hold {'ON' if args.fsdp_accum_hold else 'off'} | "
          f"pg timeout 45m", flush=True)
    # F5 lever visibility: the CONFIGURED policy (None = no cast). The RESOLVED
    # per-group dtypes only exist after lazy init — printed once at the
    # first-backward tripwire below.
    _mp = _mp_kw.get("mp_policy")
    print(f"FSDP2 dtypes: sharded master {next(_shell.parameters()).dtype} | "
          f"block policy param_dtype {getattr(_mp, 'param_dtype', None)} "
          f"reduce_dtype {getattr(_mp, 'reduce_dtype', None)} "
          f"cast_forward_inputs {getattr(_mp, 'cast_forward_inputs', None)} | "
          f"root group: no policy (fp32) | "
          f"resolved dtypes print at first backward", flush=True)


@contextlib.contextmanager
def _fsdp_eval_ctx():
    """F10 eval bracket. Eval paths call comp()/dec_trunk/lm_head DIRECTLY
    (never through the shell forward that normally unshards the root group),
    so the root is unsharded manually; per-block resharding is disabled so
    repeat forwards — generate()'s per-token loop — post no further
    collectives, making rank-divergent early stops harmless; all under
    no_grad, because a grad-enabled forward with no backward would poison
    FSDP's prefetch bookkeeping. Restores flags and reshards on exit.
    No-op outside FSDP. Rank-0-ONLY use is illegal (probe G: hangs) — every
    rank must enter/exit together."""
    if not FSDP_ON:
        yield
        return
    shell.unshard()  # root group -> plain unsharded tensors (collective)
    for _b in _FSDP_BLOCKS:
        _b.set_reshard_after_forward(False)
    try:
        with torch.no_grad():
            yield
    finally:
        for _b in _FSDP_BLOCKS:
            _b.set_reshard_after_forward(args.fsdp_reshard == "full")
            _b.reshard()
        shell.reshard()


# NOTE the "lora_" exclusion: encoder-LoRA params live on comp.encoder and
# require grad — without it they'd be silently scooped into the base group at
# --lr instead of --enc-lora-lr (found in the earlier configuration validation).
pooler_ps = [q for n, q in comp.named_parameters() if "latent_pooler" in n and q.requires_grad]
# NOTE the exclusions: "lora_" (earlier validation — else scooped into the base group at
# --lr); "encoder." under --full-ft (else the whole 1.7B lands at --lr instead
# of --enc-lr, the same silent-scoop bug one tier up).
base_ps = [q for n, q in comp.named_parameters()
           if "latent_pooler" not in n and "lora_" not in n
           and not (args.full_ft and n.startswith("encoder.")) and q.requires_grad]
groups = [{"params": base_ps, "lr": args.lr}]
if pooler_ps:
    groups.append({"params": pooler_ps, "lr": args.pooler_lr or args.lr})
if lora_ps:
    groups.append({"params": lora_ps, "lr": args.lora_lr})
if enc_lora_ps:
    groups.append({"params": enc_lora_ps, "lr": args.enc_lora_lr})
if args.full_ft:
    groups.append({"params": ft_enc_ps, "lr": args.enc_lr})
    groups.append({"params": ft_dec_ps, "lr": args.dec_lr})
if args.adam_8bit:
    assert device.startswith("cuda"), "bitsandbytes 8-bit Adam is CUDA-only"
    import bitsandbytes as bnb
    # embeddings + lm_head keep 32-bit optimizer state (the classic 8-bit
    # instability site); everything else gets 8-bit moment buffers
    mgr = bnb.optim.GlobalOptimManager.get_instance()
    stable_mods = [decoder.get_input_embeddings()]
    if getattr(decoder, "lm_head", None) is not None:
        stable_mods.append(decoder.lm_head)  # tied-to-embed double-register is harmless
    if args.full_ft:
        stable_mods.append(comp.encoder.embed_tokens)
    for m_ in stable_mods:
        mgr.register_module_override(m_, "weight", {"optim_bits": 32})
    opt = bnb.optim.AdamW8bit(groups)
    print(f"AdamW8bit: {len(stable_mods)} modules pinned to 32-bit state")
elif FSDP_ON:
    # F8: plain AdamW over the DTensor params — FSDP2 already shards optimizer
    # state by construction (each param IS its shard), so ZeRO would be
    # redundant indirection. Groups were built above from post-wrap params;
    # per-group lr survives because FQNs/param identities are unchanged (F1).
    opt = torch.optim.AdamW(groups, **({"fused": True} if args.fused_adam else {}))
elif DIST:
    # D3: multi-group at construction is supported in 2.12 (greedy global
    # bin-packing across groups; empty per-rank shards are fine). LambdaLR on
    # the wrapper propagates LRs to local shards via _sync_param_groups.
    # overlap_with_ddp is FORBIDDEN (no grad-accum / multi-group support).
    from torch.distributed.optim import ZeroRedundancyOptimizer
    opt = ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW)
else:
    opt = torch.optim.AdamW(groups)

_ALL_PS = base_ps + pooler_ps + lora_ps + enc_lora_ps + ft_enc_ps + ft_dec_ps  # telemetry union

# staged warmups: freeze everything except the named organ so it settles before
# joint training. Freezing happens AFTER the optimizer captured the params
# (grad=None -> AdamW skips them); unfrozen in the loop. --enc-warmup runs
# FIRST (repair the mask flip), then --pooler-warmup (if both are set the
# pooler phase spans (enc_warmup, enc_warmup+pooler_warmup]).
assert not (args.enc_warmup and not enc_lora_ps), "--enc-warmup requires --enc-lora-r"
assert not (args.pooler_warmup and not pooler_ps), "--pooler-warmup requires --pooling latent"
WARM_ENC_END = args.enc_warmup
WARM_POOL_END = args.enc_warmup + args.pooler_warmup


def apply_freeze(step):
    """set requires_grad per warmup phase; returns True while any warmup active"""
    if step <= WARM_ENC_END:
        active = set(map(id, enc_lora_ps))
    elif step <= WARM_POOL_END:
        active = set(map(id, pooler_ps))
    else:
        active = None
    for q in base_ps + pooler_ps + lora_ps + enc_lora_ps + ft_enc_ps + ft_dec_ps:
        q.requires_grad_(active is None or id(q) in active)
    return active is not None


def lr_lambda(s):
    if s < 100:
        return s / 100  # warmup
    if not args.lr_decay:
        return 1.0      # constant (legacy behavior; caused checkpoint jitter)
    prog = (s - 100) / max(args.steps - 100, 1)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))  # cosine to 5%


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# EMA of the projector: a running average of weights across training. The live
# weights orbit the solution (especially at constant LR); the average sits near
# the orbit's center and is usually the better artifact.
ema_state = None
ema_lora = None
ema_enc_lora = None
if args.ema:
    ema_state = {k: v.detach().clone() for k, v in comp.projector.state_dict().items()}
    if lora_ps:
        ema_lora = lora_state_dict(decoder)
    if enc_lora_ps:
        ema_enc_lora = lora_state_dict(comp.encoder)


def swap_in_ema():
    backup = {k: v.detach().clone() for k, v in comp.projector.state_dict().items()}
    comp.projector.load_state_dict(ema_state)
    lora_backup = None
    enc_backup = None
    if ema_lora is not None:
        lora_backup = lora_state_dict(decoder)
        load_lora_state_dict(decoder, ema_lora)
    if ema_enc_lora is not None:
        enc_backup = lora_state_dict(comp.encoder)
        load_lora_state_dict(comp.encoder, ema_enc_lora)
    return backup, lora_backup, enc_backup


def swap_back(backup, lora_backup, enc_backup):
    comp.projector.load_state_dict(backup)
    if lora_backup is not None:
        load_lora_state_dict(decoder, lora_backup)
    if enc_backup is not None:
        load_lora_state_dict(comp.encoder, enc_backup)

# Length-grouped sampling: batch similar-length functions together so most
# batches land in the small buckets (median function is 154 tokens) instead
# of one long function dragging three short ones up to its padded length.
import random

# Length-grouped order WITHOUT gathering a column through the shuffled view:
# datasets' formatter iterates the whole 12GB table per-batch for that (1.5h);
# pyarrow-direct column reads + composing with the shuffle permutation is ~1s
# and train[i] == raw[perm[i]]. Also assert the shuffle kept every row — a
# stale cache-*.arrow inside the dataset dir can silently shrink shuffle(seed=0)
# (observed: 2,600,000 -> 2,314,889; delete dataset-dir caches if this trips).
assert len(train) == len(train_raw), \
    f"shuffle dropped rows ({len(train)} != {len(train_raw)}) — stale dataset cache?"
_ntok_raw = train_raw.data.column("n_tokens").to_pylist()
if train._indices is not None:
    _perm = train._indices.column(0).to_pylist()
    _keys = [_ntok_raw[j] for j in _perm]
    del _perm
else:
    _keys = _ntok_raw
del _ntok_raw  # 2.6M-element python lists x 8 ranks — drop before the loop
order = sorted(range(len(train)), key=_keys.__getitem__)
del _keys


def _tb_trim(row_ids):
    """--token-budget: shrink the drawn window so sum(n_tokens) <= budget.
    Min 1 row (a single over-budget row still trains, alone). The window is
    contiguous in the length-sorted order, so the trimmed tail is the longest."""
    kept, tot = [], 0
    for rid in row_ids:
        t = _NTOK[rid]
        if kept and tot + t > args.token_budget:
            break
        kept.append(rid)
        tot += t
    return kept


rng = random.Random(0)
task_rng = random.Random(7)  # separate stream: row sampling stays identical to legacy runs
dist_rng = random.Random(13)  # separate stream: distractor draws don't perturb the others
stack_rng = random.Random(17)
qa_rng = random.Random(19)
pf_rng = random.Random(23)
hm_rng = random.Random(29)  # separate stream: hard-mine swaps don't perturb the others


# --distributed draws (D5): every control-flow decision is a pure function of
# (step, micro) — identical on all ranks with no long-lived shared stream to
# desync; content draws add the rank key (disjoint-by-seed data, D7). Stateless
# by construction => nothing here needs checkpointing on resume. int-only hash
# input: str hashes are per-process salted (PYTHONHASHSEED).
# k: 1 task, 2 pf, 3 qa-variant, 4 distractor-coin (shared);
#    5 rows, 6 stack, 7 qa-content, 8 hard-mine, 9 distractor-content,
#    10 qapre-records, 11 retr-records, 12 span-records (rank-local).
# The pregen arms (10-12) use this in LEGACY mode too: they are new behavior
# behind --qa-data, so stateless pure-fn draws (== the ws=1 distributed draws)
# cost nothing in parity and make resume checkpoint-free by construction.
def _dstream(k, step, micro, rank_key=0):
    return random.Random(hash((0xC0DEC0DE, k, rank_key, step, micro)))


def _draw_micro(step, micro):
    """Every draw + remap decision for one (step, micro) on THIS rank — the
    main loop's DIST branch and the --build-prefetch worker share this one
    implementation. Pure function of (step, micro, rank) by construction
    (every stream is a fresh _dstream; no shared state is read or written),
    which is exactly why lookahead cannot desync ranks (lever D).
    Returns (pf_pick|None, row_ids, task, qa_variant|None, mc, gm, remapped)."""
    _rk = 0 if args.dist_fixture else RANK
    _gm = RANK * args.grad_accum + micro if args.dist_fixture else micro
    # shared draws key on _gm: identical to micro in normal runs; under
    # --dist-fixture they follow the global micro so mixed-task windows
    # are grad-comparable across (ws, accum) splits (rank-varying tasks
    # are exactly what find_unused supports)
    pf_pick = None
    if PF_MIX:
        pf_pick = _dstream(2, step, _gm).choices(
            list(PF_MIX), weights=list(PF_MIX.values()))[0]
    elif args.pf_start:
        pf_pick = args.pf_start if step <= args.pf_switch else args.pf
    # +1: randrange's upper bound is exclusive — the final length-sorted
    # window start was never drawable (validation)
    s = _dstream(5, step, _gm, _rk).randrange(0, len(order) - args.batch + 1)
    row_ids = [order[s + i] for i in range(args.batch)]
    if args.token_budget:  # window is length-sorted: trimming drops the longest
        row_ids = _tb_trim(row_ids)
    task = _dstream(1, step, _gm).choices(list(MIX), weights=list(MIX.values()))[0]
    _qv = None
    if task == "qa":
        _qv = (args.force_qa_variant if args.force_qa_variant is not None
               else _dstream(3, step, _gm).random())
    # D6: any (phase, task, variant) with no trainable path remaps to recon
    # BEFORE the forward, as a shared decision — no rank can ever skip a
    # backward. plain touches only the decoder; qa<0.2 is raw text (ditto).
    # The pregen arms (qapre/retr/span) ride the same table during warmups
    # / decoder-frozen configs (brief-pinned, same condition as plain).
    _warm = WARM_POOL_END > 0 and step <= WARM_POOL_END
    _dec_side = bool(lora_ps or ft_dec_ps)
    remapped = False
    if ((task == "plain" or (task == "qa" and _qv < 0.2) or task in _QA_ARMS)
            and (_warm or not _dec_side)):
        task = "recon"
        remapped = True
    mc = {"stack_rng": _dstream(6, step, _gm, _rk),
          "qa_rng": _dstream(7, step, _gm, _rk),
          "qa_variant": (lambda v=_qv: v),
          "dist_rng": _dstream(9, step, _gm, _rk),
          "dist_coin": _dstream(4, step, _gm).random,
          "hm_rng": _dstream(8, step, _gm, _rk)}
    if QA is not None:  # pregen-arm content: rank-keyed pure-fn draws (D5/D7)
        mc.update(qapre_rng=_dstream(10, step, _gm, _rk),
                  retr_rng=_dstream(11, step, _gm, _rk),
                  span_rng=_dstream(12, step, _gm, _rk))
    return pf_pick, row_ids, task, _qv, mc, _gm, remapped


# --- lever D (--build-prefetch): ONE worker thread builds the CPU half of the
# NEXT micro (draws + dataset fetch + tokenization + AST/prose python prep)
# while the GPU runs the current one. The GPU half (H2D, comp() forward,
# assembly) and every collective (_sync_any rebuild, _sync_fwd_count stack
# truncation) stay on the MAIN thread in original order. The worker predicts
# the builders' tokenizer calls by re-running the same pure-fn draws with its
# own fresh _dstream instances; its products land in a per-micro cache served
# through the _ctok_*/_cprose/_cfacts/_cdoc/_craw seams. Prediction drift is a
# cache MISS (main thread recomputes — bit-identical result), never a
# correctness event. RNG: the worker consumes ZERO torch RNG (tokenize/arrow/
# AST are RNG-free; draws are fresh Random instances), and the single-worker
# FIFO queue keeps its python-side work in submission order by construction.
_PF_ON = args.build_prefetch
if _PF_ON:
    import queue as _queue
    import threading as _threading

    _PF_TOK = [None]   # the worker's OWN tokenizer (fast tokenizers mutate
    _PF_JOBS = _queue.Queue()   # shared padding state per call — sharing the
    _PF_RES = _queue.Queue()    # main `tok` across threads races)

    def _wtok_batch(texts, cache, offsets=False):
        kw = {"return_tensors": "pt", "padding": True, "add_special_tokens": False}
        if offsets:
            kw["return_offsets_mapping"] = True
        e = _PF_TOK[0](texts, **kw)
        cache[("tb", tuple(texts), offsets)] = e
        return e

    def _wtok_ids(text, cache):
        e = _PF_TOK[0](text, add_special_tokens=False).input_ids
        cache[("ti", text)] = e
        return e

    def _pf_prep_recon(rows, mc, cache):
        codes = rows["code"]
        want_prose = args.prose_weight != 1.0 or args.anchor_frac > 0
        want_scaf = args.span_scaffold > 0 or args.prose_weight_short is not None
        enc = _wtok_batch(codes, cache, offsets=want_prose or want_scaf)
        if want_prose:
            cache[("pr", tuple(codes))] = prose_token_mask(codes, enc.offset_mapping)
        # mirror the distractor coin + content draw (worker-local instances)
        if args.distractor_frac > 0 and mc["dist_coin"]() < args.distractor_frac:
            _wtok_ids(train[mc["dist_rng"].randrange(len(train))]["code"], cache)

    def _pf_prep_cont(rows, cache):
        want_prose = args.prose_weight != 1.0
        enc = _wtok_batch(rows["code"], cache, offsets=want_prose)
        if want_prose:
            cache[("pr", tuple(rows["code"]))] = prose_token_mask(
                rows["code"], enc.offset_mapping)

    def _pf_prep_desc(rows, cache):
        docs = []
        for c in rows["code"]:
            d = extract_docstring(c, tokenizer=_PF_TOK[0])
            cache[("dc", c)] = d
            docs.append(d)
        _wtok_batch([strip_prose_text(c) for c in rows["code"]], cache)
        _wtok_batch([d or "" for d in docs], cache)

    def _pf_prep_stack(rows, srng, cache):
        # mirror of stack_members' draw order: N, extra rows, budget trim.
        # The F4 _sync_fwd_count MIN-truncation stays on the MAIN thread; a
        # cdiv event only turns the batch entries below into misses (rare,
        # priced by the cdiv counter).
        N = srng.randint(2, args.stack_n_max)
        codes = list(rows["code"])
        while len(codes) < N:
            codes.append(train[srng.randrange(len(train))]["code"])
        codes = codes[:N]
        if args.stack_token_budget:
            kept, total = [], 0
            for c in codes:
                t = len(_wtok_ids(c, cache))
                if kept and len(kept) >= 2 and total + t > args.stack_token_budget:
                    continue
                kept.append(c)
                total += t
            codes = kept
            N = len(codes)
        names = [(m.group(1) if (m := _DEF.search(c)) else f"fn{i}")
                 for i, c in enumerate(codes)]
        _wtok_batch(codes, cache)
        for name in names:
            _wtok_ids(f"### function: {name}\n", cache)
        n_samples = max(2, min(args.batch, 40 // max(N, 1)))
        targets = srng.sample(range(N), min(n_samples, N))
        for t in targets:
            _wtok_ids(f"\nReproduce the function `{names[t]}`:\n", cache)

    def _pf_prep_qa(rows, qv, qa_rng, cache):
        codes_all = rows["code"]
        facts = []
        for c in codes_all:
            f = fn_facts(c)
            cache[("fa", c)] = f
            facts.append(f)
        keep = [i for i, f in enumerate(facts) if f is not None]
        if not keep:  # local fallback -> make_batch(rows); prep its tokenize
            want_prose = args.prose_weight != 1.0 or args.anchor_frac > 0
            want_scaf = args.span_scaffold > 0 or args.prose_weight_short is not None
            enc = _wtok_batch(codes_all, cache, offsets=want_prose or want_scaf)
            if want_prose:
                cache[("pr", tuple(codes_all))] = prose_token_mask(
                    codes_all, enc.offset_mapping)
            return
        codes = [codes_all[i] for i in keep]
        facts = [facts[i] for i in keep]
        fake_pool = sorted({c for f in facts for c in f["calls"]})
        qs, ans = [], []
        for f in facts:  # consume qa_rng in builder order (all kept, THEN cap)
            q, a = qa_pair(f, qa_rng, fake_pool)
            qs.append(q)
            ans.append(" " + a)
        _wtok_batch(ans, cache)
        if qv < 0.2:
            for c, q in zip(codes[:8], qs[:8]):
                _wtok_ids(f"{c}\n\nQ: {q}\nA:", cache)
        elif qv < 0.4:
            names = [f["name"] for f in facts[:8]]
            _wtok_batch(codes[:8], cache)
            for n_ in names:
                _wtok_ids(f"### function: {n_}\n", cache)
            for n_, q in zip(names, qs[:8]):
                _wtok_ids(f"\n\nQ: In the function `{n_}`: {q}\nA:", cache)
        else:
            _wtok_batch(codes, cache)
            for q in qs:
                _wtok_ids(f"\n\nQ: {q}\nA:", cache)

    def _pf_pick_qapre(rng_):
        """_draw_qapre minus its QA_STATS side-effect — the main-thread
        builder does the real draw (with stats) on its own identical rng."""
        if args.qapre_conc_frac > 0 and QA["qapre_conc"] \
                and rng_.random() < args.qapre_conc_frac:
            return _qa_pick_solo(QA["qapre_conc"], rng_), False
        n_solo, n_stack = len(QA["qapre_solo"]), QA["qapre_stack_n"]
        if n_stack and (not n_solo or rng_.random() < n_stack / (n_solo + n_stack)):
            return _qa_pick_stack(QA["qapre_stack"], rng_), True
        return _qa_pick_solo(QA["qapre_solo"], rng_), False

    def _pf_qa_prompt(r, cache):
        q = r["question"]
        if r.get("options"):
            q += "".join(f"\n{chr(65 + i)}) {o}" for i, o in enumerate(r["options"]))
        _wtok_ids(f"\nQ: {q}\nA:", cache)

    def _pf_prep_qa_solo(recs, cache):
        idxs = [r["row_idx"] for r in recs]
        rows = train_raw[idxs]
        cache[("rr", tuple(idxs))] = rows
        _wtok_batch(rows["code"], cache)
        for r in recs:
            _pf_qa_prompt(r, cache)
        _wtok_batch([" " + r["answer"] for r in recs], cache)

    def _pf_prep_qa_stackarm(recs, cache):
        idxs = list(recs[0]["context"]["stack_row_idxs"])
        rows = train_raw[idxs]
        cache[("rr", tuple(idxs))] = rows
        names = (["" if n is None else str(n).split(".")[-1] for n in rows["func_name"]]
                 if "func_name" in rows else
                 [(m.group(1) if (m := _DEF.search(c)) else f"fn{i}")
                  for i, c in enumerate(rows["code"])])
        _wtok_batch(rows["code"], cache)
        for n_ in names:
            _wtok_ids(f"### function: {n_}\n", cache)
        for r in recs:
            _pf_qa_prompt(r, cache)
        _wtok_batch([" " + r["answer"] for r in recs], cache)

    def _pf_build_cpu(step, micro):
        pf_pick, row_ids, task, qv, mc, gm, rm = _draw_micro(step, micro)
        rows = train[row_ids]  # the big fetch — handed over directly
        cache = {}
        out = {"step": step, "micro": micro, "row_ids": row_ids, "rows": rows,
               "cache": cache}
        try:
            if task == "recon":
                _pf_prep_recon(rows, mc, cache)
            elif task == "cont":
                _pf_prep_cont(rows, cache)
            elif task == "desc":
                _pf_prep_desc(rows, cache)
            elif task == "plain":
                _wtok_batch(rows["code"], cache)
            elif task == "stack":
                _pf_prep_stack(rows, mc["stack_rng"], cache)
            elif task == "qa":
                _pf_prep_qa(rows, qv, mc["qa_rng"], cache)
            elif task == "qapre":
                recs, st = _pf_pick_qapre(mc["qapre_rng"])
                (_pf_prep_qa_stackarm if st else _pf_prep_qa_solo)(recs, cache)
            elif task == "retr":
                _pf_prep_qa_stackarm(_qa_pick_stack(QA["retr"], mc["retr_rng"]), cache)
            elif task == "span":
                _pf_prep_qa_solo(_qa_pick_solo(QA["span"], mc["span_rng"]), cache)
        except Exception as e:
            # prep is a speed-only layer: a divergence degrades to cache
            # misses, never to a wrong batch — count it (prep_err tripwire)
            cache.clear()
            out["prep_err"] = repr(e)
        return out

    def _pf_worker():
        # (validation): the WHOLE worker body sits inside try — before
        # this fix the tokenizer construction ran outside it, so an init
        # failure killed the daemon with no result and the main thread's
        # unbounded queue get blocked that rank until the NCCL timeout, with
        # zero telemetry. Any init/loop failure now posts a fatal sentinel.
        try:
            t_ = AutoTokenizer.from_pretrained(DEC_NAME, **_HF_REV)
            if t_.pad_token is None:
                t_.pad_token = t_.eos_token
            _PF_TOK[0] = t_
        except Exception as e:
            _PF_RES.put({"step": -1, "micro": -1, "fatal": e})
            return
        while True:
            job = _PF_JOBS.get()
            if job is None:
                return
            try:
                _PF_RES.put(_pf_build_cpu(*job))
            except Exception as e:  # e.g. dataset fetch died — surface on main
                _PF_RES.put({"step": job[0], "micro": job[1], "fatal": e})

    _PF_THREAD = _threading.Thread(target=_pf_worker, daemon=True, name="build-prefetch")
    _PF_THREAD.start()

    def _pf_submit(step, micro):
        if step <= args.steps:
            _PF_JOBS.put((step, micro))

    def _pf_next_coord(step, micro):
        return (step, micro + 1) if micro + 1 < args.grad_accum else (step + 1, 0)

    _PF_TAKE_TIMEOUT_S = float(os.environ.get("PF_TAKE_TIMEOUT_S", "600"))

    def _pf_take(step, micro):
        # : bounded take with a liveness check — a dead worker (or a
        # wedged fetch) aborts THIS rank loudly inside the collective timeout
        # instead of hanging it silently past 45 min at $32/hr.
        t0_ = time.time()
        while True:
            try:
                res = _PF_RES.get(timeout=15.0)
                break
            except _queue.Empty:
                if not _PF_THREAD.is_alive():
                    raise RuntimeError(
                        f"[rank {RANK}] --build-prefetch worker thread DIED "
                        "without posting a result — see fatal sentinel path; "
                        "aborting loudly ()") from None
                if time.time() - t0_ > _PF_TAKE_TIMEOUT_S:
                    raise RuntimeError(
                        f"[rank {RANK}] --build-prefetch worker alive but "
                        f"silent for {_PF_TAKE_TIMEOUT_S:.0f}s at (step {step}, "
                        f"micro {micro}) — wedged fetch? aborting loudly "
                        "()") from None
        w = (time.time() - t0_) * 1000
        PF_STATS["wait_ms"] = 0.98 * (PF_STATS["wait_ms"] or w) + 0.02 * w
        if "fatal" in res:
            raise RuntimeError(
                f"--build-prefetch worker failed at (step {res['step']}, "
                f"micro {res['micro']}; -1/-1 = worker init)") from res["fatal"]
        assert (res["step"], res["micro"]) == (step, micro), \
            f"prefetch pipeline desync: got {(res['step'], res['micro'])}, " \
            f"expected {(step, micro)} — submission-order bug"
        if "prep_err" in res:
            PF_STATS["prep_err"] += 1
        return res

# --- FSDP checkpointing (FSDP DESIGN F9) --------------------------------------
# Resume state rides torch.distributed.checkpoint (DCP): each rank writes its
# own shard in parallel — the DDP path's 30-minute rank-0 consolidate
# (the legacy path) dies with the ZeRO path. Eval/verdict ARTIFACTS keep their
# plain gathered-.pt formats (helpers below) so compressor/ft_load.py and the
# §4 evaluators need no changes.
_DCP_FUT = None       # at most one async save in flight (F9 discipline)
_DCP_PENDING = None   # (step, tmp_dir) of the in-flight save
CKPT_DIR = run_dir / "ckpt"
if FSDP_ON:
    import shutil

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict, get_state_dict, set_state_dict)
    from torch.distributed.checkpoint.stateful import Stateful

    # DEDICATED gloo group for checkpoint I/O: async_save's write thread runs
    # its own collectives — on the DEFAULT group they interleave with the next
    # training step's collectives and corrupt the stream (observed: gloo
    # 'op.nread == op.preamble.nbytes' SIGABRT on the first post-save step).
    # A separate CPU group gives the checkpoint thread its own channel; on the
    # node the default pg is cpu:gloo,cuda:nccl and this carves the gloo half.
    from datetime import timedelta as _td
    _DCP_PG = dist.new_group(backend="gloo", timeout=_td(minutes=45))

    def _fsdp_full_sd(module):
        """Full fp32 state dict of an ORIGINAL module on rank 0 ({} on others).
        Collective — ALL ranks must call. FQN key sets are byte-identical to
        the single-process trainer because the gather never sees the shell
        (danger #14: shell FQNs would break ft_load's strict loads)."""
        return get_model_state_dict(module, options=StateDictOptions(
            full_state_dict=True, cpu_offload=True))

    def _fsdp_decoder_sd():
        sd = _fsdp_full_sd(decoder)
        if IS_MAIN and "lm_head.weight" not in sd and "model.embed_tokens.weight" in sd:
            # tied weights: the DCP gather dedups the shared param to one FQN;
            # decoder.state_dict() (the ft_load/eval contract) carries BOTH
            # keys. Same tensor object -> torch.save stores one copy.
            sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
        return sd

    def _fsdp_gather_dict(d):
        """dict of (possibly DTensor) tensors -> full plain CPU tensors, on
        every rank (all-gather per entry; sized for projector/EMA/LoRA dicts).
        Collective — ALL ranks must call, same dict key order everywhere."""
        return {k: (v.full_tensor().cpu() if isinstance(v, _DT) else v.detach().cpu())
                for k, v in d.items()}

    class _AppState(Stateful):
        """One DCP payload: model+optimizer via get/set_state_dict (sharded
        DTensors) + scheduler/py_rng/step/EMA + the resume contract fields.
        set_state_dict is only legal before backward() or after step() — the
        resume call site (before the loop) satisfies it."""

        def __init__(self):
            self.step = 0

        def state_dict(self):
            msd, osd = get_state_dict(shell, opt)
            sd = {"model": msd, "opt": osd, "sched": sched.state_dict(),
                  "py_rng": rng.getstate(), "step": self.step,
                  "args_hash": ARGS_HASH, "world_size": WORLD,
                  "format": "fsdp2-v1"}
            for k, v in (("ema", ema_state), ("ema_lora", ema_lora),
                         ("ema_enc_lora", ema_enc_lora)):
                if v is not None:
                    sd[k] = v  # sharded DTensor dicts: DCP shards the write
            return sd

        def load_state_dict(self, sd):
            # belt-and-braces: the sidecar precheck already validated these
            # before any tensor was touched; a torn/foreign dir dies here
            assert sd.get("format") == "fsdp2-v1", \
                f"DCP checkpoint format {sd.get('format')!r} != fsdp2-v1 — refusing"
            assert sd.get("args_hash") == ARGS_HASH, \
                f"resume args_hash mismatch (ckpt {sd.get('args_hash')} != run {ARGS_HASH})"
            set_state_dict(shell, opt, model_state_dict=sd["model"],
                           optim_state_dict=sd["opt"])
            sched.load_state_dict(sd["sched"])
            rng.setstate(sd["py_rng"])
            # ema tensors were loaded IN PLACE (state_dict returned the live
            # DTensor objects) — nothing further to restore for them
            self.step = sd["step"]

    _app = _AppState()

    def _dcp_wait():
        """Finalize the in-flight async save: block on the write, then rank 0
        renames tmp -> stepN, drops the sidecar meta, flips the LATEST pointer
        (atomic), prunes older complete checkpoints. All-rank (barriers) —
        every rank must call at the same point."""
        global _DCP_FUT, _DCP_PENDING
        if _DCP_FUT is None:
            return
        _DCP_FUT.result()
        dist.barrier()  # every rank's shard is on disk
        stepN, tmp = _DCP_PENDING
        if IS_MAIN:
            final = CKPT_DIR / f"step{stepN}"
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
            (final / "app_meta.json").write_text(json.dumps(
                {"args_hash": ARGS_HASH, "world_size": WORLD, "step": stepN,
                 "format": "fsdp2-v1"}))
            _lt = CKPT_DIR / "LATEST.tmp"
            _lt.write_text(f"step{stepN}")
            _lt.rename(CKPT_DIR / "LATEST")  # pointer flip = commit
            for d_ in CKPT_DIR.glob("step*"):
                if d_.name not in (f"step{stepN}", f"step{stepN}.tmp") and d_.is_dir():
                    shutil.rmtree(d_, ignore_errors=True)
        dist.barrier()
        _DCP_FUT = None
        _DCP_PENDING = None

    def _dcp_poll():
        """Opportunistic finalize: flip LATEST as soon as EVERY rank's write is
        done (shared decision — .done() may differ across ranks). All-rank."""
        if _DCP_FUT is None:
            return
        if not _sync_any(not _DCP_FUT.done()):  # nobody still writing
            _dcp_wait()

    _DCP_ASYNC_DEAD = False  # sticky: once the WORLD agrees async failed, stay sync

    def _dcp_save(stepN):
        """Kick off the async sharded save. async_save STAGES the state to CPU
        synchronously before returning (DefaultStager, non-async staging), so
        training may mutate params immediately after; the write itself runs in
        a thread and is committed by _dcp_poll/_dcp_wait. Falls back to a
        synchronous dcp.save if async_save refuses (it is self-described
        experimental). All-rank.

        (validation): the fallback decision is COLLECTIVE. Before this
        fix each rank caught its own exception rank-locally — one rank could
        enter sync dcp.save + _dcp_wait's barriers while peers held futures
        and marched on, mismatching collective order (hang). Now every rank
        reports its local outcome into one allreduce on the DEFAULT group
        (never _DCP_PG — the async write thread owns that channel) and:
        all-failed -> everyone falls back (sticky); none-failed -> async
        proceeds; ASYMMETRIC -> every rank aborts loudly (the ranks with live
        futures cannot cancel them, so any continuation risks a collective-
        order mismatch — a loud unanimous abort beats a 45-min silent hang)."""
        global _DCP_FUT, _DCP_PENDING, _DCP_ASYNC_DEAD
        _dcp_wait()  # at most one in flight
        _app.step = stepN
        tmp = CKPT_DIR / f"step{stepN}.tmp"
        if IS_MAIN:
            CKPT_DIR.mkdir(exist_ok=True)
            if tmp.exists():
                shutil.rmtree(tmp)
        dist.barrier()
        # Checkpoint staging needs contiguous GPU-side shard copies. Release
        # cached blocks first so the writer has room at peak training memory.
        # The operation is negligible at the 2000-step save cadence.
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        _fail, _err, _fut = False, None, None
        _t_async_fail = os.environ.get("FSDP_TEST_ASYNC_FAIL", "")
        if not _DCP_ASYNC_DEAD:
            try:
                if _t_async_fail == "all" or (_t_async_fail == "r0" and IS_MAIN):
                    # test only: simulate async_save refusing on
                    # all ranks (fallback path) / rank 0 only (asymmetry abort)
                    raise RuntimeError("FSDP_TEST_ASYNC_FAIL injected")
                _fut = dcp.async_save({"app": _app}, checkpoint_id=str(tmp),
                                      process_group=_DCP_PG)
            except Exception as _e:  # e.g. no CPU backend in the pg
                _fail, _err = True, _e
        # collective agreement (default group): (any_fail, all_fail) in one op
        _t = torch.tensor([int(_fail or _DCP_ASYNC_DEAD),
                           -int(_fail or _DCP_ASYNC_DEAD)],
                          dtype=torch.int64, device=device)
        dist.all_reduce(_t, op=dist.ReduceOp.MIN)  # -> (min=all, -max=-any)
        _all_fail, _any_fail = bool(int(_t[0])), bool(int(-_t[1]))
        if _any_fail and not _all_fail:
            raise RuntimeError(
                f"[rank {RANK}] step {stepN}: dcp.async_save failed on SOME "
                f"ranks only (local fail={_fail}: {_err!r}) — asymmetric "
                "async state is unrecoverable; aborting all ranks loudly "
                "(). Resume from the last committed LATEST.")
        if _all_fail:
            if not _DCP_ASYNC_DEAD:
                print(f"dcp.async_save unavailable on ALL ranks "
                      f"({type(_err).__name__ if _err else '?'}: {_err}) — "
                      "falling back to synchronous dcp.save (sticky)", flush=True)
            _DCP_ASYNC_DEAD = True
            dcp.save({"app": _app}, checkpoint_id=str(tmp), process_group=_DCP_PG)
            _DCP_FUT, _DCP_PENDING = _SyncDone(), (stepN, tmp)
            _dcp_wait()
            return
        _DCP_FUT = _fut
        _DCP_PENDING = (stepN, tmp)
        if os.environ.get("FSDP_TEST_KILL_AFTER_SAVE") == "1":
            # test mode: die with the write in flight — the checkpoint
            # must be either complete (previous LATEST) or absent, never torn
            os._exit(17)

    class _SyncDone:
        def done(self):
            return True

        def result(self):
            return None


# --- resume -------------------------------------------------------------------
start_step = 0
state_path = run_dir / ("full_state_latest.pt" if args.full_ft else "state_latest.pt")
if FSDP_ON and args.resume:
    # F9: resume state lives in the DCP dir; full_state_latest.pt is an
    # ARTIFACT (no opt/sched/rng) and must never be half-restored (danger #15)
    _latest_p = CKPT_DIR / "LATEST"
    if not _latest_p.exists():
        if state_path.exists():
            sys.exit(f"FATAL: --resume --fsdp: {state_path} exists but no DCP "
                     f"checkpoint ({CKPT_DIR}/LATEST). A DDP/legacy-era or "
                     "artifact-only state cannot be resumed into the FSDP "
                     "trainer (format fsdp2-v1) — it has no sharded opt state.")
        sys.exit(f"--resume: {CKPT_DIR}/LATEST missing — refusing to silently start fresh")
    _cdir = CKPT_DIR / _latest_p.read_text().strip()
    _meta = json.loads((_cdir / "app_meta.json").read_text())
    # precheck BEFORE any tensor is loaded (dcp.load mutates in place)
    assert _meta.get("format") == "fsdp2-v1", \
        f"resume: {_cdir} format {_meta.get('format')!r} != fsdp2-v1 — refusing"
    assert _meta.get("args_hash") == ARGS_HASH, \
        f"resume args_hash mismatch (ckpt {_meta.get('args_hash')} != run {ARGS_HASH}): " \
        "byte-identical flags are REQUIRED to resume — a changed --steps alone " \
        "silently reshapes the cosine schedule"
    assert _meta.get("world_size") == WORLD, \
        f"resume world-size lock: checkpoint was written at ws={_meta.get('world_size')}, " \
        f"this run is ws={WORLD}. DCP could reshard, but ARGS_HASH pins world_size " \
        "by policy (rank-keyed data streams + eff batch) — resume at the saved ws."
    dcp.load({"app": _app}, checkpoint_id=str(_cdir), process_group=_DCP_PG)
    start_step = _app.step
    print(f"resumed from step {start_step} (DCP {_cdir.name})")
elif args.resume and not state_path.exists():
    # a silent fresh start under --resume mixes step-1 records into an
    # existing run's logs and eventually overwrites its state (validation)
    sys.exit(f"--resume: {state_path} missing — refusing to silently start fresh")
if not FSDP_ON and args.resume and state_path.exists():
    # distributed: every rank loads the full file CPU-side; ZeRO keeps only its
    # shard after load_state_dict (which mutates its argument — never reuse st)
    st = torch.load(state_path, map_location="cpu" if DIST else device)
    # args_hash is now saved unconditionally; checked whenever present (older
    # checkpoints predate the key). Single-card resumes were exempt and a
    # changed --steps silently reshaped the LambdaLR schedule (validation).
    if st.get("args_hash") is not None or DIST:
        assert st.get("args_hash") == ARGS_HASH, \
            f"resume args_hash mismatch (ckpt {st.get('args_hash')} != run {ARGS_HASH}): " \
            "byte-identical flags are REQUIRED to resume — a changed --steps alone " \
            "silently reshapes the cosine schedule"
    if args.full_ft:
        comp.encoder.load_state_dict(st["enc"])
        decoder.load_state_dict(st["dec"])
        print("full-ft weights restored (encoder + decoder)")
    comp.projector.load_state_dict(st["projector"])
    # projector EMA must be restored too — before validation (validation) it
    # silently restarted from the pre-resume init, so post-resume EMA exports
    # were hybrids (fresh projector EMA + resumed LoRA EMAs).
    # Checkpoint tensors are CPU-side under DIST (map_location) — land them on
    # the live tensors' device/dtype or the first in-place EMA update crashes
    # with a CPU/CUDA mismatch (validation).
    if ema_state is not None and st.get("ema") is not None:
        ema_state = {k: v.to(ema_state[k].device, ema_state[k].dtype)
                     for k, v in st["ema"].items()}
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    rng.setstate(st["py_rng"])
    start_step = st["step"]
    if lora_ps and st.get("lora") is not None:
        load_lora_state_dict(decoder, st["lora"])
        if ema_lora is not None and st.get("ema_lora") is not None:
            ema_lora = {k: v.to(ema_lora[k].device, ema_lora[k].dtype)
                        for k, v in st["ema_lora"].items()}
    if enc_lora_ps and st.get("enc_lora") is not None:
        load_lora_state_dict(comp.encoder, st["enc_lora"])
        if ema_enc_lora is not None and st.get("ema_enc_lora") is not None:
            ema_enc_lora = {k: v.to(ema_enc_lora[k].device, ema_enc_lora[k].dtype)
                            for k, v in st["ema_enc_lora"].items()}
    del st  # dense enc+dec+opt tensors held per rank (CPU under DIST, the
    # live device single-card) — retaining it multiplies a ~68GB checkpoint
    # across 8 ranks and can OOM a single-card resume (validation)
    print(f"resumed from step {start_step}")

# --- resume telemetry truncation (validation, fixed recipe) ---------------
# The jsonls append forever while resume rolls training back to the restored
# checkpoint, so a crash AFTER logging steps beyond the last save left
# abandoned-lineage records (duplicate step numbers) that K2/K3 monitor reads
# could consume (its next() took the FIRST — the dead lineage). On every
# resume, rank 0 atomically rewrites each telemetry file keeping only records
# with step <= start_step; the replayed steps then append fresh, single-
# lineage records. Atomic tmp+rename: a crash mid-trim leaves the old file
# intact (records are only ever dropped, never mutated, so the monitor can
# never read a torn line that parsed before). NOTE: cumulative rank-local
# counters (task_count/QA_STATS/FSDP_STATS) still restart at zero on resume —
# documented in fixed recipe; they are monotone diagnostics, not criteria.
if args.resume and start_step > 0 and IS_MAIN:
    def _trim_jsonl(p, max_step):
        if not p.exists():
            return 0
        kept, dropped = [], 0
        for _l in open(p):
            try:
                _s = json.loads(_l).get("step")
            except json.JSONDecodeError:
                dropped += 1  # torn tail line of the crashed run: drop
                continue
            if _s is not None and _s > max_step:
                dropped += 1
            else:
                kept.append(_l)
        if dropped:
            _t = p.with_suffix(p.suffix + ".trim")
            with open(_t, "w") as _f:
                _f.writelines(kept)
            _t.rename(p)
        return dropped
    metrics_f.close()  # append fd would keep writing the unlinked inode
    _n_trim = sum(_trim_jsonl(run_dir / f, start_step)
                  for f in ("metrics.jsonl", "steps.jsonl", "gen_samples.jsonl"))
    metrics_f = open(run_dir / "metrics.jsonl", "a")
    if _n_trim:
        print(f"resume: trimmed {_n_trim} abandoned-lineage telemetry records "
              f"(step > {start_step}) — single-lineage files restored ()")

# --- distributed wrap (D1/D2): construct BEFORE apply_freeze — DDP only manages
# params that require grad at construction time, and warmup phases unfreeze
# theirs later (frozen phases show up as globally-unused: grad stays None,
# AdamW skips them identically on every rank). Construct outside autocast.
# (--fsdp: the shell was wrapped with fully_shard BEFORE the param groups —
# see the FSDP2 wrap block above; this DDP ctor is the preserved fallback.)
if DIST and not FSDP_ON:
    _shell = DDPShell(comp, dec_trunk, lm_head)
    shell = torch.nn.parallel.DistributedDataParallel(
        _shell,
        device_ids=[LOCAL_RANK] if device.startswith("cuda") else None,
        find_unused_parameters=True,   # task arms have disjoint param coverage (map §3)
        broadcast_buffers=False,       # HF buffers static; init_sync still checks parity once
        gradient_as_bucket_view=True)  # ~23GB/rank peak saved on 5.7B fp32 grads
    # startup banner (D10) + duplicated-data estimate (D7: disjoint-by-seed, not
    # partitioned — overlap must be provably noise-level at 3M+ rows)
    _nd = WORLD * args.grad_accum
    print(f"DDP+ZeRO1: world {WORLD} | batch {args.batch}/rank x accum {args.grad_accum} "
          f"= eff {WORLD * args.batch * args.grad_accum} | backend {dist.get_backend()} | "
          f"find_unused on | bucket 25MB | grad-as-bucket-view on | pg timeout 45m | "
          f"managed {sum(q.numel() for q in _shell.parameters() if q.requires_grad) / 1e6:.1f}M "
          f"params | E[overlapping window pairs/step] "
          f"{_nd * (_nd - 1) / 2 * (2 * args.batch - 1) / max(len(order) - args.batch, 1):.4f}",
          flush=True)

apply_freeze(start_step + 1)  # correct phase whether fresh or resumed mid-warmup
if args.full_ft:
    # transformers 5.x gates gradient checkpointing on module.training. A
    # decoder left in eval() makes --dec-grad-ckpt a silent no-op and will run
    # out of memory on long batches. Qwen3 has no dropout, so train() changes
    # memory behavior but not numerics. Eval and generation temporarily switch
    # back to eval(), and each training step restores train mode. The encoder
    # needs the same treatment for --enc-grad-ckpt.
    decoder.train()
    comp.train()
elif enc_lora_ps or args.enc_grad_ckpt:
    # same transformers-5.x gate for non-full-FT encoder-side configs: the
    # encoder child loads in eval mode, so enc grad-ckpt was inert until the
    # first post-eval comp.train() — and under DDP only rank 0 runs evaluate,
    # leaving other ranks in eval forever (validation). Qwen3 has zero
    # dropout: train() changes memory behavior only, never numerics.
    comp.train()

print(f"training: {args.steps} steps, batch {args.batch}, lr {args.lr}")
t0 = time.time()
task_ema = {}  # per-task running loss (0.98 EMA) — which objective is moving
task_count = {}  # realized task mix (sampled != configured on short runs)
build_t_ema, compute_t_ema = [0.0], [0.0]  # batch-build vs fwd/bwd wall split
steps_f = open(run_dir / "steps.jsonl", "a") if IS_MAIN else None  # fine-grained log, every 10 steps


def organ_norms():
    """weight-space diagnostics per organ — the earlier configuration 'is the pooler alive?'
    question, answered live instead of by checkpoint archaeology.

    FSDP (F8): params are sharded DTensors; float(norm()) would read a
    per-shard partial and shrink the specified 'organs' field (danger
    #9). The few touched params are materialized via full_tensor — a
    collective, so under FSDP this runs on ALL ranks (eval boundary)."""
    def _t(x):
        return x.detach().full_tensor() if FSDP_ON and isinstance(x, _DT) else x
    d = {}
    if pooler_ps:
        lp = comp.projector.latent_pooler
        d["pool_o"] = float(_t(lp.o_proj.weight).norm())
        d["pool_gate"] = float(_t(lp.imp_gate).mean())
        d["pool_slope_max"] = float(_t(lp.log_slope).exp().max())
    if lora_ps:
        d["lora_B"] = sum(float(_t(q).norm()) for n, q in decoder.named_parameters()
                          if "lora_B" in n)
    if enc_lora_ps:
        d["enc_B"] = sum(float(_t(q).norm()) for n, q in comp.encoder.named_parameters()
                         if "lora_B" in n)
    return {k: round(v, 3) for k, v in d.items()}


def gpu_health():
    """power/temp/clock snapshot at evals — host-quality drift on the record"""
    if not device.startswith("cuda"):  # `cuda:N` under --distributed (map #16)
        return {}
    try:
        # one CSV line per GPU on a multi-GPU host — parse this rank's line
        # (a whole-output split() mangled field 3 and the
        # bare except turned every eval's snapshot into {} on the 8x node)
        lines = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5
        ).stdout.strip().splitlines()
        q = lines[min(LOCAL_RANK if DIST else 0, len(lines) - 1)].split(", ")
        return {"gpu_w": float(q[0]), "gpu_temp": int(q[1]), "gpu_mhz": int(q[2]), "gpu_util": int(q[3])}
    except Exception:
        return {}


def heartbeat(step, note="", done=False):
    """one-line liveness file: remote check-ins read this instead of tailing logs.

    Atomic write + epoch timestamp ('t'): a torn read or a monitor in a
    different timezone must not fake or miss a hang (validation). 'done' is
    an explicit lifecycle flag written only after final exports (or a clean
    halt) — step==of alone fires BEFORE final gen/save/export and let the
    most expensive final failure look complete."""
    _hb_tmp = run_dir / "heartbeat.tmp"
    with open(_hb_tmp, "w") as hb:
        hb.write(json.dumps({"step": step, "of": args.steps, "phase": cur_phase(step),
                             "s_per_step": round((time.time() - t0) / max(step, 1), 2),
                             "t": round(time.time(), 1), "done": done,
                             "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note}))
    _hb_tmp.rename(run_dir / "heartbeat")


def _asave(obj, path):
    """atomic torch.save (tmp + rename): a crash mid-save must not tear a
    verdict artifact like projector_ema.pt (validation); the resume state
    already saved this way"""
    _t = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, _t)
    _t.rename(path)


def grad_norms():
    """L2 grad norm per trained organ (grads persist until next zero_grad).

    FSDP (F8): grads are SHARDED — the per-rank partial would silently shrink
    the specified 'g' field by ~sqrt(world) and mis-fire K-criteria
    calibrated on earlier configurations (danger #9). Squared sums are therefore summed
    across ranks in ONE collective; under FSDP every rank must call this at
    the same point (call sites: %10 and %50 boundaries, all-rank)."""
    names, sqs = [], []
    for name, ps in (("proj", base_ps), ("pool", pooler_ps),
                     ("lora", lora_ps), ("enc", enc_lora_ps),
                     ("ftenc", ft_enc_ps), ("ftdec", ft_dec_ps)):
        if not ps:
            continue
        sq = 0.0
        for q in ps:
            if q.grad is None:
                continue
            g = q.grad
            if FSDP_ON and isinstance(g, _DT):
                g = g._local_tensor  # local shard; no implicit collective
            sq += float(g.pow(2).sum())
        names.append(name)
        sqs.append(sq)
    if FSDP_ON and names:
        t = torch.tensor(sqs, dtype=torch.float64, device=device)
        dist.all_reduce(t)  # SUM of per-rank squared partials = global
        sqs = t.tolist()
    return {k: v ** 0.5 for k, v in zip(names, sqs)}


def cur_phase(step):
    if step <= WARM_ENC_END:
        return "enc-warmup"
    if step <= WARM_POOL_END:
        return "pooler-warmup"
    return "joint"


if IS_MAIN and start_step > 0:
    # 4B: refresh the heartbeat before the loop — the on-disk one is stale
    # (its done flag reflects the PREVIOUS run's exit), so a monitor polling
    # during the minutes until the first %50 step would false-TRIP a resume.
    # This follows cur_phase's definition because heartbeat() calls it; the
    # path runs only when resuming.
    heartbeat(start_step, note="resume")


# --distributed telemetry (D10), reset never — monotone counters + EMAs:
# [0] D6 remap count (no-op tripwire: 0 forever in a joint-phase full-ft run),
# [1] micro count, [2] synced-backward ms EMA, [3] no_sync-backward ms EMA
# (2 minus 3 ~ allreduce wait: the comm-bound leading indicator),
# [4] first synced backward ms, kept OUT of the EMA — it doubles as an implicit
# barrier and eats the slowest rank's startup lag + reducer lazy init
DDP_STATS = [0, 0, 0.0, 0.0, -1.0]


def _boundary_reduce(local_loss):
    """Collective telemetry at print/eval boundaries only (D10): mean loss +
    per-task EMAs, summed task counts, and a grad health read. DDP: cross-rank
    spread of sum|g| — every rank holds identical averaged grads, so spread ~0
    is the silent-corruption tripwire. FSDP (F8/F10): that invariant is FALSE
    BY CONSTRUCTION (grads are sharded; spread is expected), so grad_spread is
    RETIRED and replaced by grad_l1 — the SUM of per-shard L1 partials, i.e.
    the true global L1 (finite and > 0 is the liveness read) — plus the
    non-None-grad count asymmetry, which must still be 0 (reduce-scatter
    materializes sharded grads on every rank). Must be called by ALL ranks."""
    keys = sorted(set(MIX) | {"recon"})  # D6 remaps can synthesize recon outside MIX
    v = torch.zeros(3 * len(keys) + 3, dtype=torch.float64)
    for i, k in enumerate(keys):
        v[3 * i] = task_ema.get(k, 0.0)
        v[3 * i + 1] = float(k in task_ema)
        v[3 * i + 2] = float(task_count.get(k, 0))
    v[-3] = local_loss

    def _l1(q):
        g = q.grad
        if g is None:
            return 0.0
        if FSDP_ON and isinstance(g, _DT):
            g = g._local_tensor  # local shard, no implicit collective
        return float(g.abs().sum())

    v[-2] = sum(_l1(q) for q in _ALL_PS)
    v[-1] = float(sum(1 for q in _ALL_PS if q.grad is not None))
    # NCCL-backed groups reject CPU tensors — collectives ride the run device
    v = v.to(device)
    gath = [torch.zeros_like(v) for _ in range(WORLD)]
    dist.all_gather(gath, v)
    g = torch.stack(gath).cpu()
    te = {k: float(g[:, 3 * i].sum() / g[:, 3 * i + 1].sum())
          for i, k in enumerate(keys) if float(g[:, 3 * i + 1].sum()) > 0}
    tc = {k: int(g[:, 3 * i + 2].sum()) for i, k in enumerate(keys)
          if float(g[:, 3 * i + 2].sum()) > 0}
    fp, nz = g[:, -2], g[:, -1]
    asym = int(nz.max() - nz.min())
    if FSDP_ON:
        return {"loss": float(g[:, -3].mean()), "task_ema": te, "task_count": tc,
                "grad_l1": float(fp.sum()), "grad_nz_asym": asym}
    spread = (float((fp.max() - fp.min()) / max(float(fp.abs().max()), 1e-12))
              if asym == 0 else -1.0)
    return {"loss": float(g[:, -3].mean()), "task_ema": te, "task_count": tc,
            "grad_spread": spread, "grad_nz_asym": asym}


_FSDP_CHECK_GRADS = FSDP_ON  # arm the first-backward sharded/fp32 tripwire
_FSDP_HOLD = FSDP_ON and args.fsdp_accum_hold  # accum-window param residency
# fixed recipe "ANY non-finite loss anywhere", enforced per MICRO (validation
# ): the grad-norm gate below catches non-finite GRADS, but a non-finite
# forward loss with finite/zero derivatives (or a transient the %10 sampling
# misses) was outside it. Device-side |= per micro costs zero host syncs; the
# flag is read once per optimizer step at the pre-step gate (where a host
# sync already happens for the grad norm) and aborts BEFORE opt.step().
_LOSS_NONFIN = torch.zeros((), dtype=torch.bool, device=device)
# --torch-profile window: parsed/validated on ALL ranks (identical args ->
# identical failure), trace collected on rank 0 only. Stop fires after the
# step's optimizer work and BEFORE the eval block, so step B's eval never
# lands in the trace; interior eval boundaries would, hence the warning.
_TPROF = None
_TPROF_START = _TPROF_END = -1
if args.torch_profile:
    _a, _, _b = args.torch_profile.partition(":")
    _TPROF_START, _TPROF_END = int(_a), int(_b or _a)
    assert 1 <= _TPROF_START <= _TPROF_END, \
        f"--torch-profile {args.torch_profile!r}: need 1 <= A <= B"
    assert _TPROF_END - _TPROF_START + 1 <= 10, \
        f"--torch-profile {args.torch_profile!r}: window > 10 steps"
    assert _TPROF_START > start_step, \
        f"--torch-profile {args.torch_profile!r}: window starts at or before " \
        f"the resume point (step {start_step}) — it would never trigger"
    assert _TPROF_END <= args.steps \
        and (args.halt_after_step is None or _TPROF_END <= args.halt_after_step), \
        f"--torch-profile {args.torch_profile!r}: window must close before the " \
        f"run ends (steps={args.steps}, halt={args.halt_after_step})"
    # a kineto trace still live at interpreter exit SIGABRTs in libc++ mutex
    # teardown (observed  --dump-grads exits mid-window). Backstop: stop
    # (and discard — an interrupted window is not a valid trace) on any exit
    # path that skips the in-loop close.
    import atexit

    def _tprof_abort():
        global _TPROF
        if _TPROF is not None:
            _TPROF.stop()
            print("torch-profile: run ended inside the trace window — trace "
                  "discarded", flush=True)
            _TPROF = None
    atexit.register(_tprof_abort)
    if IS_MAIN:
        if any(s % args.eval_every == 0 for s in range(_TPROF_START, _TPROF_END)):
            print("torch-profile WARNING: window interior spans an eval "
                  "boundary — eval work will dominate the trace", flush=True)
        print(f"torch-profile: armed for steps {_TPROF_START}..{_TPROF_END} (rank 0)",
              flush=True)
if _PF_ON:
    _pf_submit(start_step + 1, 0)  # lever D bootstrap: one job in flight
for step in range(start_step + 1, args.steps + 1):
    _t_step0 = time.time()  # true optimizer-step wall (excludes eval blocks)
    if step == _TPROF_START and IS_MAIN:
        from torch.profiler import ProfilerActivity, profile as _kineto_profile
        _TPROF = _kineto_profile(
            activities=[ProfilerActivity.CPU]
            + ([ProfilerActivity.CUDA] if device == "cuda" else []),
            record_shapes=True)
        _TPROF.start()
        print(f"torch-profile: trace started at step {step}", flush=True)
    if args.margin_weight > 0:
        # ramped: early margins reflect an unconverged readout, not real ties
        MARGIN_K = args.margin_weight * min(1.0, step / max(args.margin_ramp, 1))
    if args.rank_loss > 0:
        RANK_K = args.rank_loss * min(1.0, step / max(args.rank_ramp, 1))
    if args.hard_mine > 0 and step % 500 == 0 and len(HARD_EMA) > 8192:
        for rid, _ in sorted(HARD_EMA.items(), key=lambda kv: kv[1])[: len(HARD_EMA) - 8192]:
            del HARD_EMA[rid]
    if step in (WARM_ENC_END + 1, WARM_POOL_END + 1) and step > 1 and WARM_POOL_END > 0:
        apply_freeze(step)
        _FSDP_CHECK_GRADS = FSDP_ON  # re-arm: unfreeze is #186998's trigger
        phase = "pooler warmup" if step == WARM_ENC_END + 1 and step <= WARM_POOL_END else "joint"
        print(f"warmup phase change @ step {step}: -> {phase}", flush=True)
    opt.zero_grad()
    _stepped = False
    for _micro in range(args.grad_accum):
        if _CKPT_PROBE:
            _pc0, _pc1 = _CKPT_PROBE_CNT  # per-micro layer-forward deltas (an earlier configuration)
        if DIST:
            # D5/D7: shared pure-fn control-flow draws, rank-keyed content
            # draws — one implementation in _draw_micro (shared verbatim with
            # the --build-prefetch worker; comments live there)
            _tb = time.time()
            _pf_pick, row_ids, task, _qv, mc, _gm, _rm = _draw_micro(step, _micro)
            if _pf_pick is not None:
                comp.pooling_factor = _pf_pick
            if _rm:
                DDP_STATS[0] += 1
            if _PF_ON:
                # lever D handshake: take THIS micro's CPU-half products, then
                # immediately queue the next micro so the worker overlaps the
                # GPU work below. rows handover is exact (same row_ids, same
                # deterministic fetch); the cache is served via the _c* seams.
                _pfr = _pf_take(step, _micro)
                _pf_submit(*_pf_next_coord(step, _micro))
                if _pfr["row_ids"] == row_ids:
                    rows = _pfr["rows"]
                    PF_STATS["rows_hit"] += 1
                else:  # can only mean a prep/draw drift — priced, never wrong
                    rows = train[row_ids]
                    PF_STATS["rows_miss"] += 1
                _PF_ACTIVE = _pfr["cache"]
            else:
                rows = train[row_ids]
        else:
            if PF_MIX:  # mixed-ratio: one model, all ratios — each micro-batch samples its pf
                comp.pooling_factor = pf_rng.choices(list(PF_MIX), weights=list(PF_MIX.values()))[0]
            elif args.pf_start:  # ratio curriculum: learn verbatim habits while capacity is cheap
                comp.pooling_factor = args.pf_start if step <= args.pf_switch else args.pf

            _tb = time.time()
            s = rng.randrange(0, len(order) - args.batch + 1)  # +1: see DIST path (r2 #27)
            row_ids = [order[s + i] for i in range(args.batch)]
            if args.token_budget:  # window is length-sorted: trimming drops the longest
                row_ids = _tb_trim(row_ids)
            rows = train[row_ids]

            task = task_rng.choices(list(MIX), weights=list(MIX.values()))[0]
            if WARM_POOL_END > 0 and step <= WARM_POOL_END and task == "plain":
                # plain = raw-text LM, no vectors: with decoder-LoRA frozen (either
                # warmup phase) there is NO trainable path (backward crashes) — and its
                # purpose (LoRA anti-forgetting) is moot while LoRA is frozen.
                task = "recon"
            elif (task in _QA_ARMS
                  and ((WARM_POOL_END > 0 and step <= WARM_POOL_END)
                       or not (lora_ps or ft_dec_ps))):
                # D6-mirror for the pregen arms: same remap condition as the
                # distributed table, so legacy and ws=1 --distributed behave alike
                task = "recon"
            mc = {"stack_rng": stack_rng, "qa_rng": qa_rng, "qa_variant": qa_rng.random,
                  "dist_rng": dist_rng, "dist_coin": dist_rng.random, "hm_rng": hm_rng}
            if QA is not None:  # pregen-arm content: same pure-fn draws as ws=1 DIST
                mc.update(qapre_rng=_dstream(10, step, _micro, 0),
                          retr_rng=_dstream(11, step, _micro, 0),
                          span_rng=_dstream(12, step, _micro, 0))
        if FSDP_ON:
            # F2 — THE load-bearing change: NO no_sync. Every micro-batch's
            # backward reduce-scatters its grads, which then accumulate on the
            # SHARDED fp32 DTensor (~1.7GB/rank) instead of piling up
            # unsharded (13.4GB — the leg-3 OOM). FSDP's no_sync equivalent
            # (set_requires_gradient_sync(False)) accumulates UNSHARDED and
            # would reproduce the OOM exactly; the negative control is the tripwire.
            if _FSDP_HOLD and _micro == 0:
                # --fsdp-accum-hold window start: no group reshards after
                # forward OR backward until opt.step — one all-gather per group
                # per WINDOW instead of fwd+bwd per micro. While held, module
                # attrs point at the UNSHARDED copies (grad=None); the
                # reduce-scattered grads land on the SHARDED DTensor params the
                # post-wrap ps lists/optimizer hold, so clip/step/telemetry are
                # unaffected (probe: analysis, validation). Re-armed +
                # explicitly resharded right after opt.step.
                for _g in _FSDP_BLOCKS:
                    _g.set_reshard_after_forward(False)
                    _g.set_reshard_after_backward(False)
                shell.set_reshard_after_backward(False, recurse=False)
                FSDP_STATS["hold_windows"] += 1
            loss = shell(task, rows, row_ids, step, mc)
            if not loss.requires_grad:
                # a rank skipping a backward = divergent collective schedule =
                # hang minutes later at $32/hr; die loud and immediately (D6)
                raise RuntimeError(f"[rank {RANK}] step {step} task {task}: micro-batch "
                                   "has no trainable path — the D6 remap table is wrong")
            _tbw = time.time()
            (loss / args.grad_accum).backward()
            _ms = (time.time() - _tbw) * 1000
            if DDP_STATS[4] < 0:
                DDP_STATS[4] = _ms  # first backward: startup outlier, logged apart
            else:
                FSDP_STATS["bwd_ms"] = 0.98 * (FSDP_STATS["bwd_ms"] or _ms) + 0.02 * _ms
            DDP_STATS[1] += 1
            if _FSDP_CHECK_GRADS and _micro == 0:
                _FSDP_CHECK_GRADS = False
                # first-backward tripwires — re-armed after every warmup phase
                # change, because #186998's trigger is a module whose first
                # forward ran with requires_grad=False: (a) grads SHARDED —
                # a plain-tensor grad means an unwrapped/stale param (danger
                # #7/#12); (b) grads fp32 — catches a silently-bf16 reduce
                # (#186998 / unset reduce_dtype, danger #8) empirically rather
                # than by introspection.
                _n_sh = 0
                for _q in _ALL_PS:
                    if _q.grad is None:
                        continue
                    assert isinstance(_q.grad, _DT), \
                        f"[rank {RANK}] non-DTensor grad after FSDP backward " \
                        f"(shape {tuple(_q.grad.shape)}) — stale/unwrapped param"
                    assert _q.grad._local_tensor.numel() < max(_q.grad.numel(), 2) \
                        or WORLD == 1, \
                        f"[rank {RANK}] grad not sharded: local " \
                        f"{_q.grad._local_tensor.numel()} of {_q.grad.numel()}"
                    assert _q.grad.dtype == torch.float32, \
                        f"[rank {RANK}] grad dtype {_q.grad.dtype} != fp32 — " \
                        "gradient reduction ran in low precision (F5/#186998)"
                    _n_sh += 1
                assert _n_sh > 0, f"[rank {RANK}] first backward produced no grads"
                if args.fsdp_param_dtype == "bfloat16":
                    # (c) the fp32-grad assert above CANNOT catch a bf16
                    # REDUCTION: foreach_reduce casts the reduce output back to
                    # the fp32 master dtype AFTER reducing (verified 2.12.1
                    # _fsdp_collectives.py; probe validation — grads arrive
                    # fp32 even when the sum ran in 8 mantissa bits). So the
                    # real tripwire introspects each group's RESOLVED reduce
                    # dtype: None here means "reduce in the bf16 grad dtype" —
                    # both the unset-reduce_dtype trap and #186998's lazy trap
                    # (a group whose first forward ran all-frozen resolves
                    # None permanently) land as None. Groups with no trainable
                    # params post no reduction and are skipped. BLOCK groups
                    # only: the root group carries no mp_policy (fp32 island,
                    # see the wrap comment), so its None is the legitimate
                    # "no cast" and its grads reduce fp32 natively.
                    _rd_n = 0
                    for _g in _FSDP_BLOCKS:
                        _pg = _g._get_fsdp_state()._fsdp_param_group
                        if _pg is None or not any(
                                _fp.sharded_param.requires_grad
                                for _fp in _pg.fsdp_params):
                            continue
                        _rd = getattr(_pg, "_reduce_dtype", None)
                        assert _rd is torch.float32, \
                            f"[rank {RANK}] FSDP group " \
                            f"'{getattr(_pg, '_module_fqn', '?')}' resolved " \
                            f"reduce dtype {_rd} != fp32 under " \
                            "--fsdp-param-dtype bfloat16 — gradients are " \
                            "reducing in bf16 (F5/#186998); aborting"
                        _rd_n += 1
                    print(f"FSDP2 resolved dtypes @ first backward: block "
                          f"all-gather bf16, reduce fp32 (introspected across "
                          f"{_rd_n} trainable block groups); root group fp32 "
                          f"(no cast); sharded grads fp32", flush=True)
            _stepped = True
        elif DIST:
            _sync = _micro == args.grad_accum - 1
            # D4: forward INSIDE no_sync for non-final micro-steps
            with (contextlib.nullcontext() if _sync else shell.no_sync()):
                loss = shell(task, rows, row_ids, step, mc)
                if not loss.requires_grad:
                    # one rank skipping a backward = reducer desync = NCCL hang
                    # minutes later at $32/hr; die loud and immediately (D6)
                    raise RuntimeError(f"[rank {RANK}] step {step} task {task}: micro-batch "
                                       "has no trainable path — the D6 remap table is wrong")
                _tbw = time.time()
                (loss / args.grad_accum).backward()
            _ms = (time.time() - _tbw) * 1000
            if _sync and DDP_STATS[4] < 0:
                DDP_STATS[4] = _ms  # startup outlier, logged separately
            else:
                _i = 2 if _sync else 3
                DDP_STATS[_i] = 0.98 * (DDP_STATS[_i] or _ms) + 0.02 * _ms
            DDP_STATS[1] += 1
            _stepped = True
        else:
            loss = micro_loss(task, rows, row_ids, step, mc)
            if loss.requires_grad:  # false only if a batch has no trainable path (warmup edge)
                (loss / args.grad_accum).backward()
                _stepped = True
        _LOSS_NONFIN |= ~torch.isfinite(loss.detach())  # device op, no sync ()
        if _PF_ON:
            _PF_ACTIVE = None  # cache is per-micro; eval paths must never see it
        if _CKPT_PROBE:
            # an earlier configuration probe line: layer-0 forward invocations this micro (fwd-only
            # = 1x, checkpointed fwd+recompute = 2x) next to the gate decision
            print(f"CKPTPROBE step={step} micro={_micro} "
                  f"dec_l0={_CKPT_PROBE_CNT[0] - _pc0} "
                  f"enc_l0={_CKPT_PROBE_CNT[1] - _pc1} "
                  f"dec_gate={CKPT_LAST['dec']} enc_gate={CKPT_LAST['enc']} "
                  f"dec_tot={CKPT_LAST_TOT['dec']} enc_tot={CKPT_LAST_TOT['enc']} "
                  f"dec_call={CKPT_LAST_CALL['dec']} enc_call={CKPT_LAST_CALL['enc']}",
                  flush=True)
        _tc = MICRO_TC[0]
        build_t_ema[0] = 0.98 * (build_t_ema[0] or _tc - _tb) + 0.02 * (_tc - _tb)
        compute_t_ema[0] = 0.98 * (compute_t_ema[0] or time.time() - _tc) + 0.02 * (time.time() - _tc)
        task_count[task] = task_count.get(task, 0) + 1
    if _stepped:
        # under DDP every rank holds identical averaged grads after the synced
        # backward, so the plain union clip stays rank-identical (D4). With
        # clip off, max_norm=inf measures without modifying. The finiteness
        # gate runs BEFORE opt.step(): one NaN/inf grad otherwise mutates the
        # replicated weights and poisons the next checkpoint minutes before
        # any sampled telemetry could catch it (validation; fixed recipe
        # "ANY non-finite anywhere" is now enforced pre-mutation). NaN grads
        # are rank-identical post-allreduce, so every rank raises together.
        _dump_raw = args.dump_grads_raw is not None and step == start_step + 1
        _dump_post = args.dump_grads is not None and step == start_step + 1
        if (_dump_raw or _dump_post) and _FSDP_HOLD:
            # held window: live named_parameters() would yield the UNSHARDED
            # copies (grad=None — grads live on the sharded params). Reshard
            # first so the dump walk sees the sharded DTensors + grads; safe,
            # the process exits after the dump(s). Done ONCE, before either
            # dump and before the clip (clip reads the same sharded grads
            # either way).
            for _g in _FSDP_BLOCKS:
                _g.reshard()
            shell.reshard()

        def _grad_dump(path):
            """gather + save {name: grad} — FSDP gathers FULL tensors (all-rank
            collective, rank 0 saves); DDP/legacy rank-0 saves directly."""
            _named = ([("comp." + n, q) for n, q in comp.named_parameters()]
                      + [("dec." + n, q) for n, q in decoder.named_parameters()])
            if FSDP_ON:
                _dump = {}
                for n, q in _named:
                    if not q.requires_grad:
                        continue
                    g_ = q.grad
                    _dump[n] = (g_.full_tensor().cpu() if isinstance(g_, _DT)
                                else g_.detach().clone()) if g_ is not None else None
                if IS_MAIN:
                    torch.save(_dump, path)
                    print(f"grads dumped -> {path}", flush=True)
            elif IS_MAIN:  # DDP: post-allreduce grads are rank-identical; legacy trivial
                torch.save({n: (q.grad.detach().clone() if q.grad is not None else None)
                            for n, q in _named if q.requires_grad}, path)
                print(f"grads dumped -> {path}", flush=True)

        if _dump_raw:
            # /: UNCLIPPED gradients — the magnitudes production
            # applies when --grad-clip is 0. Dumped BEFORE clip_grad_norm_
            # touches anything, so R2's unclipped arms measure the actual
            # accumulation contract, not clip-normalized directions.
            _grad_dump(args.dump_grads_raw)
        _gn_total = torch.nn.utils.clip_grad_norm_(
            base_ps + pooler_ps + lora_ps + enc_lora_ps + ft_enc_ps + ft_dec_ps,
            args.grad_clip if args.grad_clip > 0 else float("inf"))
        if FSDP_ON and isinstance(_gn_total, _DT):
            # F8: over DTensor grads clip_grad_norm_ computes the GLOBAL norm
            # (probe F: Replicate placement, rank-identical) but returns a
            # DTensor — float()/isfinite() on it would read a partial or
            # trigger an implicit collective. Materialize explicitly (all
            # ranks reach this together).
            _gn_total = _gn_total.full_tensor()
        if not torch.isfinite(_gn_total):
            raise RuntimeError(
                f"[rank {RANK}] step {step}: non-finite grad norm ({float(_gn_total)}) "
                "before opt.step() — aborting pre-mutation (fixed recipe). "
                "Diagnose, then resume from the last saved state.")
        if bool(_LOSS_NONFIN):
            # : a micro of THIS window produced a non-finite forward loss
            # (its grads may be finite/zero — the norm gate above can miss it).
            # Same pre-mutation contract: abort before opt.step(). The flag is
            # rank-local; every rank that saw one raises; a single-rank raise
            # still kills the job via the collective timeout, loudly.
            raise RuntimeError(
                f"[rank {RANK}] step {step}: non-finite LOSS in >=1 micro of "
                "this window — aborting pre-mutation (fixed recipe, per-micro "
                "enforcement, validation).")
        if _dump_post:
            # post-clip dump (legacy R2 semantics: sharded grads gathered to
            # FULL tensors so dumps compare across world sizes)
            _grad_dump(args.dump_grads)
        if _dump_raw or _dump_post:
            if DIST:
                dist.barrier()
            sys.exit(0)
        if FSDP_ON:
            _dcp_poll()  # commit a finished async checkpoint (all-rank, cheap)
        opt.step()
        if _FSDP_HOLD:
            # opt.step() mutated the SHARDED masters, so every held unsharded
            # copy is now STALE — the next forward's unshard() would early-
            # return and silently train on pre-step weights. Re-arm the normal
            # reshard flags (eval/DCP/EMA paths below see baseline behavior)
            # and explicitly reshard EVERY group; this must run BEFORE the EMA
            # block (state_dict() walks live module attrs). The assert is the
            # fail-loud tripwire: a missed reshard is a silent-corruption bug,
            # never a memory detail (negative control proves it fires).
            if not args.fsdp_hold_no_reshard:  # TEST ONLY skip (B-negative)
                for _g in _FSDP_BLOCKS:
                    _g.set_reshard_after_forward(args.fsdp_reshard == "full")
                    _g.set_reshard_after_backward(True)
                    _g.reshard()
                shell.set_reshard_after_backward(True, recurse=False)
                shell.reshard()
            for _g in _FSDP_BLOCKS + [shell]:
                _pg = _g._get_fsdp_state()._fsdp_param_group
                assert _pg is None or not _pg.is_unsharded, \
                    f"[rank {RANK}] step {step}: FSDP group " \
                    f"'{getattr(_pg, '_module_fqn', '?')}' still UNSHARDED " \
                    "after opt.step under --fsdp-accum-hold — its stale " \
                    "gathered copy would silently serve pre-step weights " \
                    "(missed post-step reshard)"
    sched.step()
    # D8: EMA is rank-0-only under DDP — ZeRO's step() broadcasts every updated
    # shard before returning, so rank 0's post-step params equal everyone's.
    # FSDP (F10): EMA is SHARDED and ALL-RANK — rank 0 holds only 1/world of
    # each param, so the rank-0-only rule would ship a 1/8-populated
    # projector_ema.pt, the §4 HEADLINE artifact (danger #13). Every rank EMAs
    # its own shard elementwise; 1/world the arithmetic, gathered only at save.
    if ema_state is not None and (IS_MAIN or FSDP_ON):
        with torch.no_grad():
            for k, v in comp.projector.state_dict().items():
                ema_state[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)
            if ema_lora is not None:
                for k, v in lora_state_dict(decoder).items():
                    ema_lora[k].mul_(args.ema).add_(v, alpha=1 - args.ema)
            if ema_enc_lora is not None:
                for k, v in lora_state_dict(comp.encoder).items():
                    ema_enc_lora[k].mul_(args.ema).add_(v, alpha=1 - args.ema)

    task_ema[task] = 0.98 * task_ema.get(task, loss.item()) + 0.02 * loss.item()
    if _TPROF is not None and step == _TPROF_END:
        if device == "cuda":
            torch.cuda.synchronize()  # flush in-flight kernels into the trace
        _TPROF.stop()
        _tr = run_dir / f"torch_trace_{_TPROF_START}_{_TPROF_END}.json.gz"
        _TPROF.export_chrome_trace(str(_tr))
        _sort = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
        print(_TPROF.key_averages().table(sort_by=_sort, row_limit=25), flush=True)
        print(f"torch-profile: chrome trace -> {_tr} "
              f"({_tr.stat().st_size / 1e6:.1f} MB)", flush=True)
        _TPROF = None
    if step % 10 == 0 and (IS_MAIN or FSDP_ON):  # per-step record stays rank-local (D10)
        # FSDP: grad_norms is a COLLECTIVE (sharded grads, F8) — every rank
        # computes it here; only rank 0 writes the record below
        if FSDP_ON:
            _nrf_cov_drain()  # F10 layer-2: all-rank local drain ()
        _g10 = grad_norms()
    if step % 10 == 0 and IS_MAIN:
        srec = {"step": step, "loss": round(loss.item(), 4), "task": task,
                "pf": comp.pooling_factor, "lr": sched.get_last_lr()[0],
                "g": {k: round(v, 3) for k, v in _g10.items()},
                "t_build": round(build_t_ema[0], 3),
                "t_compute": round(compute_t_ema[0], 3),
                # RAW wall of THIS optimizer step (all micros + opt + EMA;
                # no eval). t_build/t_compute are per-MICRO EMAs — a 30-step
                # profile reads throughput from t_step, not from those
                "t_step": round(time.time() - _t_step0, 3)}
        if args.rank_loss > 0 and RANK_STATS[0] > 0:
            # running since last step-line reset — diagnostic trend lines
            srec["rank"] = {"k": round(RANK_K, 3),
                            "act": round(RANK_STATS[1] / RANK_STATS[0], 4),
                            "act_n": RANK_STATS[1],
                            "hinge": round(RANK_STATS[2] / max(RANK_STATS[1], 1), 3),
                            "m_np": round(RANK_STATS[3] / RANK_STATS[0], 3),
                            "m_conf": round(RANK_STATS[4] / max(RANK_STATS[5], 1), 3)}
            if args.rank_active_norm:
                srec["rank"]["anorm"] = 1
            if args.rank_recon_only:
                srec["rank"]["rgate"] = 1  # stats are recon-batches-only under the gate
        if args.hard_mine > 0:
            srec["hm"] = {"pool": len(HARD_EMA), "sw": HARD_STATS[0],
                          "att": round(HARD_STATS[2] / max(HARD_STATS[1], 1), 2)}
        if args.token_budget:
            # realized batch under the budget since the last write: rows, real
            # (unpadded) decoder tokens, micro-batches — divide by n for per-micro
            if TELEM_BATCH:
                # boundary drain: the batched path's ONLY tb host sync
                # (rank-0-local like the legacy reset; int64 -> exact ints)
                TB_STATS[0], TB_STATS[1], TB_STATS[2] = TB_ACC.tolist()
                TB_ACC.zero_()
            srec["tb"] = {"rows": TB_STATS[0], "tok": TB_STATS[1], "n": TB_STATS[2]}
            TB_STATS[0] = TB_STATS[1] = TB_STATS[2] = 0
        if args.build_prefetch:
            # lever D telemetry (monotone): rows/tok hit rates are the leading
            # indicator; tok_hit==0 or prep_err>0 = dead/drifted prep; wait_ms
            # EMA = residual main-thread build stall. Key says _r0 explicitly:
            # PF_STATS accumulates on every rank but ONLY rank 0's is written
            # — cache misses/prep errors on ranks 1..7 are invisible here
            # (fatal worker failures abort loudly on their own rank instead;
            # documented-rank-0-only decision).
            srec["pfetch_r0"] = {"rows_hit": PF_STATS["rows_hit"],
                              "rows_miss": PF_STATS["rows_miss"],
                              "tok_hit": PF_STATS["tok_hit"],
                              "tok_miss": PF_STATS["tok_miss"],
                              "prep_err": PF_STATS["prep_err"],
                              "wait_ms": round(PF_STATS["wait_ms"], 1)}
        if CKPT_N > 0:
            # lever E telemetry (reset per write): micros gated on/off per
            # model — both zero on one side = threshold never mixes = miscal
            srec["ckpt"] = dict(CKPT_STATS)
            for _k in CKPT_STATS:
                CKPT_STATS[_k] = 0
        if args.pad_to_max:
            # lever F telemetry (reset per write): realized encoder pad widths;
            # nm16 (not a multiple of 16) is the no-op tripwire, must stay 0
            srec["pad"] = {"n": PAD_STATS[0], "w_min": PAD_STATS[1],
                           "w_max": PAD_STATS[2], "nm16": PAD_STATS[3]}
            PAD_STATS[0] = PAD_STATS[1] = PAD_STATS[2] = PAD_STATS[3] = 0
        if FSDP_ON:
            # F10 telemetry: bwd_ms is the single backward EMA (there is no
            # unsynced backward any more — DDP's bwd_local_ms is retired);
            # cdiv/cdiv_max = F4 count-sync leading indicator + truncation
            # cost; rebuild/qafb = the shared-decision syncs; nrf = the
            # not-run-forward warning tripwire (must stay 0)
            srec["fsdp"] = {"bwd_ms": round(FSDP_STATS["bwd_ms"], 1),
                            "cdiv": FSDP_STATS["count_div_events"],
                            "cdiv_max": FSDP_STATS["count_div_max"],
                            "rebuild": FSDP_STATS["rebuild_sync"],
                            "qafb": FSDP_STATS["qa_fallback_sync"],
                            "nrf": FSDP_STATS["not_run_forward_warns"],
                            "remap": DDP_STATS[0]}
            if args.fsdp_accum_hold:  # key present iff the lever is on
                srec["fsdp"]["hold_w"] = FSDP_STATS["hold_windows"]
        elif DIST:
            srec["ddp"] = {"bwd_sync_ms": round(DDP_STATS[2], 1),
                           "bwd_local_ms": round(DDP_STATS[3], 1),
                           "remap": DDP_STATS[0]}
        steps_f.write(json.dumps(srec) + "\n")
        steps_f.flush()
        # 4A F-4: heartbeat rides the %10 record, not the %50 print — at 50
        # steps between beats, s/step >= ~18s crosses the monitor's 15-min
        # staleness bar and false-trips a healthy-but-slow run
        heartbeat(step)
    if step % 50 == 0:
        _l50 = loss.item()
        if DIST:  # boundary-only collective (D10); every rank must reach it.
            # tensor lives on the run device: NCCL groups reject CPU tensors
            _t50 = torch.tensor([_l50], dtype=torch.float64, device=device)
            dist.all_reduce(_t50)
            _l50 = float(_t50) / WORLD
        extra = f"  lr {sched.get_last_lr()[0]:.1e}"
        gn = grad_norms()
        if gn:
            extra += "  |g| " + " ".join(f"{k}:{v:.2f}" for k, v in gn.items())
        extra += f"  build/compute {build_t_ema[0]:.2f}/{compute_t_ema[0]:.2f}s"
        if SURP is not None:
            f_, m_ = SURP_HITS
            extra += f"  surp-hits {f_}/{f_ + m_}"
            if f_ == 0 and m_ > 50:
                extra += "  !! SURPRISAL COLUMN NOT MATCHING — weighting is a NO-OP !!"
        if TELEM_BATCH:
            # boundary drain (the batched path's only margin host syncs, 2 per
            # 50 steps, all-rank like the legacy reset); values differ from the
            # legacy path only by fp32 device accumulation order
            MARGIN_STATS[0] = float(MARGIN_ACC_F)
            MARGIN_STATS[1], MARGIN_STATS[2], MARGIN_STATS[3] = MARGIN_ACC_I.tolist()
            MARGIN_ACC_F.zero_()
            MARGIN_ACC_I.zero_()
        if MARGIN_STATS[1] > 0:
            extra += (f"  margin {MARGIN_STATS[0]/MARGIN_STATS[1]:.2f} "
                      f"neg {100*MARGIN_STATS[2]/MARGIN_STATS[1]:.2f}% "
                      f"<tau {100*MARGIN_STATS[3]/MARGIN_STATS[1]:.1f}%")
            MARGIN_STATS[0] = 0.0
            MARGIN_STATS[1] = MARGIN_STATS[2] = MARGIN_STATS[3] = 0
        if RANK_STATS[0] > 0:
            # act = hinge-active fraction (no-op tripwire: ~0 dead, ~1 miscal);
            # hinge = mean over active (leading indicator, should FALL);
            # m-np = mean non-prose margin (should RISE);
            # m-conf = mean confident margin (the narrowing detector: HOLD)
            extra += (f"  rank k{RANK_K:.2f} act {100*RANK_STATS[1]/RANK_STATS[0]:.1f}% "
                      f"hinge {RANK_STATS[2]/max(RANK_STATS[1],1):.2f} "
                      f"m-np {RANK_STATS[3]/RANK_STATS[0]:.2f} "
                      f"m-conf {RANK_STATS[4]/max(RANK_STATS[5],1):.2f}")
            if args.rank_active_norm:
                extra += f" [anorm actN {RANK_STATS[1]}]"
            RANK_STATS[0] = RANK_STATS[1] = RANK_STATS[5] = 0
            RANK_STATS[2] = RANK_STATS[3] = RANK_STATS[4] = 0.0
        if QA is not None and QA_STATS["leak"] > 0:
            extra += f"  !! QA HELD-OUT LEAK: {QA_STATS['leak']} sampled — MUST be 0 !!"
        if args.hard_mine > 0:
            extra += f"  hm pool {len(HARD_EMA)} sw {HARD_STATS[0]}"
            if step > 2000 and not HARD_EMA:
                extra += "  !! HARD-MINE POOL EMPTY — attribution not landing, mining is a NO-OP !!"
            HARD_STATS[0] = HARD_STATS[1] = 0
            HARD_STATS[2] = 0.0
        if FSDP_ON:
            # MARGIN/RANK/HARD segments above are rank-local — say so on the line
            _un = sum(1 for q in _ALL_PS if q.grad is None)  # frozen/unused this window
            extra += (f"  fsdp bwd {FSDP_STATS['bwd_ms']:.0f}ms "
                      f"cdiv {FSDP_STATS['count_div_events']}"
                      f"/max{FSDP_STATS['count_div_max']} "
                      f"rebuild {FSDP_STATS['rebuild_sync']} "
                      f"nrf {FSDP_STATS['not_run_forward_warns']} "
                      f"remap {DDP_STATS[0]} no-grad-p {_un} [stats r0/{WORLD}]")
        elif DIST:
            # MARGIN/RANK/HARD segments above are rank-local — say so on the line
            _un = sum(1 for q in _ALL_PS if q.grad is None)  # globally-unused this window
            extra += (f"  ddp bwd sync/local {DDP_STATS[2]:.0f}/{DDP_STATS[3]:.0f}ms "
                      f"remap {DDP_STATS[0]} unused-p {_un} [stats r0/{WORLD}]")
        print(f"step {step:5d}  train {_l50:.3f} [{task}]  "
              f"({(time.time()-t0)/step:.2f}s/step){extra}", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        _ec_ms = _ec_free_gb = None
        if device.startswith("cuda"):
            # Release cached CUDA blocks at evaluation boundaries so allocator
            # fragmentation cannot accumulate during long training runs. This
            # is rank-local, has no collective effect, and runs before the
            # boundary reduction.
            _t_ec = time.time()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _ec_ms = round((time.time() - _t_ec) * 1000, 1)
            _ec_free_gb = round(torch.cuda.mem_get_info()[0] / 2**30, 1)
        _red = None
        if DIST:
            # ALL collectives happen here, in fixed order on every rank, BEFORE
            # any rank-0-only I/O: telemetry reduce, then (DDP only) shard
            # consolidation for any state save this boundary performs (D8)
            _red = _boundary_reduce(loss.item())
            if FSDP_ON:
                if _red["grad_nz_asym"] != 0:
                    # reduce-scatter materializes sharded grads on EVERY rank
                    # for globally-used params — asymmetry means a rank lost a
                    # backward path (F10; grad_spread itself is retired: shard
                    # values legitimately differ, danger #11)
                    print(f"!! FSDP GRAD DESYNC: non-None-grad count asymmetry "
                          f"{_red['grad_nz_asym']} across ranks !!", flush=True)
            elif _red["grad_spread"] > 1e-6 or _red["grad_nz_asym"] != 0:
                # nz_asym should be structurally impossible: DDP materializes
                # grads on every rank for globally-used params
                print(f"!! DDP GRAD DESYNC: cross-rank grad fingerprint spread "
                      f"{_red['grad_spread']:.2e} nz-asym {_red['grad_nz_asym']} !!",
                      flush=True)
            if not FSDP_ON and ((not args.full_ft) or step % args.full_save_every == 0
                                or step == args.steps):
                # consolidate measured ~30 min at 3.4B/8 ranks (the legacy path,
                # validation): stamp the heartbeat first or the 15-min staleness
                # rule false-trips on every full save. FSDP replaces this with
                # the sharded DCP save below (F9) — the consolidate is dead.
                if IS_MAIN:
                    heartbeat(step, note="saving")
                opt.consolidate_state_dict(to=0)
        if IS_MAIN or FSDP_ON:
            # D8/D9 (DDP/legacy): eval + every write on rank 0. FSDP (F10):
            # eval COMPUTE runs on EVERY rank — a rank-0-only forward through
            # sharded modules posts unmatched all-gathers and hangs (probe G).
            # All ranks compute the same numbers on the same fixed data; the
            # WRITES stay rank-0 (non-main print is already shimmed to no-op).
            with _fsdp_eval_ctx():
                v, nc, cp = evaluate()
            captured = (nc - v) / max(nc - cp, 1e-6)
            rec = {"step": step, "train": _red["loss"] if DIST else loss.item(),
                   "pf": comp.pooling_factor, "val_vec": v,
                   "val_noctx": nc, "val_copy": cp, "captured": captured,
                   "lr": sched.get_last_lr()[0], "phase": cur_phase(step),
                   "task_loss": {k: round(x, 4) for k, x in
                                 (_red["task_ema"] if DIST else task_ema).items()},
                   "task_count": _red["task_count"] if DIST else dict(task_count),
                   "organs": organ_norms(), **gpu_health(),
                   "t_build": round(build_t_ema[0], 3), "t_compute": round(compute_t_ema[0], 3),
                   "elapsed_min": round((time.time() - t0) / 60, 1)}
            if SURP is not None:
                rec["surp_hits"] = list(SURP_HITS)
            if QA is not None:
                # rank-local (like MARGIN/RANK stats — r0/W under --distributed):
                # cumulative source counts + span answer-length histogram, per-source
                # loss EMAs, and the held-out leak tripwire (must stay 0)
                rec["qa"] = {"leak": QA_STATS["leak"],
                             "conc_n": QA_STATS.get("conc_n", 0),
                             "src_n": dict(QA_STATS["src_n"]),
                             "src_ema": {k: round(v, 4) for k, v in sorted(QA_SRC_EMA.items())},
                             "span_len": {str(k): v for k, v in sorted(QA_STATS["span_len"].items())}}
            if args.span_scaffold > 0 and args.pooling == "latent":
                with _fsdp_eval_ctx():
                    rec["scaffold_drift"] = drift_probe()  # per-kind |mass - proposal|
            if device.startswith("cuda"):  # `cuda:N` under --distributed (map #16)
                rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            if _ec_ms is not None:
                # defrag telemetry (leading indicator: ec_free_gb trending down
                # across evals = fragmentation the empty_cache isn't clearing)
                rec["ec_ms"], rec["ec_free_gb"] = _ec_ms, _ec_free_gb
            if FSDP_ON:
                rec["fsdp"] = {"grad_l1": _red["grad_l1"],
                               "grad_nz_asym": _red["grad_nz_asym"],
                               "bwd_ms": round(FSDP_STATS["bwd_ms"], 1),
                               "first_bwd_ms": round(DDP_STATS[4], 1),
                               "cdiv": FSDP_STATS["count_div_events"],
                               "cdiv_max": FSDP_STATS["count_div_max"],
                               "rebuild": FSDP_STATS["rebuild_sync"],
                               "qafb": FSDP_STATS["qa_fallback_sync"],
                               "nrf": FSDP_STATS["not_run_forward_warns"],
                               "remap_n": DDP_STATS[0], "micro_n": DDP_STATS[1]}
                if args.fsdp_accum_hold:  # key present iff the lever is on
                    rec["fsdp"]["hold_w"] = FSDP_STATS["hold_windows"]
            elif DIST:
                rec["ddp"] = {"grad_spread": _red["grad_spread"],
                              "grad_nz_asym": _red["grad_nz_asym"],
                              "bwd_sync_ms": round(DDP_STATS[2], 1),
                              "bwd_local_ms": round(DDP_STATS[3], 1),
                              "first_sync_ms": round(DDP_STATS[4], 1),
                              "remap_n": DDP_STATS[0], "micro_n": DDP_STATS[1]}
            if IS_MAIN:
                heartbeat(step, note="eval")
            line = (f"  eval@{step}: vectors {v:.3f} | floor(no-ctx) {nc:.3f} | ceiling(copy) {cp:.3f} "
                    f"| captured {100*captured:.1f}%")
            # capacity tripwire: val loss at every OTHER mixed ratio, each eval.
            # floor/ceiling dedup (unconditional, exact): those two do not
            # depend on pf — computed once in the base evaluate() above and
            # reused here. EVAL_NO_DEDUP=1 re-runs the old recompute-and-
            # discard path (validation's byte-identity A/B only).
            _wf = os.environ.get("EVAL_NO_DEDUP") == "1"
            for apf in [x for x in GEN_PFS if x != args.pf]:
                with _fsdp_eval_ctx():
                    av, _, _ = evaluate(pf=apf, with_floor=_wf)
                rec[f"val_vec_pf{apf}"] = av
                rec[f"captured_pf{apf}"] = (nc - av) / max(nc - cp, 1e-6)
                line += f" | vec@{apf}x {av:.3f}"
            if os.environ.get("FSDP_TEST_EVAL_DUMP") == "1":
                # validation only: per-rank eval values on disk so the
                # replicated-eval "identical across ranks" claim is checkable
                with open(run_dir / f"evals_rank{RANK}.jsonl", "a") as _ef:
                    _ef.write(json.dumps(
                        {"step": step, "v": v, "nc": nc, "cp": cp,
                         **{k: rec[k] for k in rec if k.startswith("val_vec_pf")}}) + "\n")
            if step % args.gen_every == 0 or step == args.steps:
                # EMA swap-in mutates live weights. DDP: rank-0-local, swap-back
                # before the post-eval barrier — no collective may run while EMA
                # weights are in (danger #9). FSDP: EVERY rank swaps its own
                # sharded EMA (identical decision), so weights stay consistent
                # across ranks even inside the eval bracket's collectives; the
                # swap itself runs OUTSIDE the bracket (sharded state).
                bak = lbak = ebak = None
                if ema_state is not None:
                    bak, lbak, ebak = swap_in_ema()
                tag = "GEN(ema)" if ema_state is not None else "GEN"
                _pf = comp.pooling_factor
                for gpf in GEN_PFS:
                    comp.pooling_factor = gpf
                    with _fsdp_eval_ctx():
                        n_exact, n_code_exact, avg_sim = gen_eval(dump_step=step)
                    sfx = "" if gpf == args.pf else f"_pf{gpf}"
                    rec.update({f"gen_exact{sfx}": n_exact, f"gen_code_exact{sfx}": n_code_exact,
                                "gen_canaries": len(canaries), f"gen_sim{sfx}": avg_sim})
                    line += (f" | {tag}@{gpf}x {n_exact}/{len(canaries)} exact, "
                             f"{n_code_exact}/{len(canaries)} code-exact, sim {100*avg_sim:.1f}%")
                if PROBE_META is not None:
                    with _fsdp_eval_ctx():
                        rp = rank_probe(pf=args.pf)
                    rec["rank_probe"] = rp
                    line += " | probe " + " ".join(
                        f"{k[:5]}:{v['wrong']}w/{v['act']}a/{v['n']}"
                        for k, v in rp.items()
                        if k in ("ident_first", "ident_rep", "string", "punct"))
                comp.pooling_factor = _pf
                if bak is not None:
                    swap_back(bak, lbak, ebak)
            print(line, flush=True)
            if IS_MAIN:
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
            if FSDP_ON:
                # F9(b): eval/verdict artifacts gathered to FULL plain tensors
                # with key sets byte-identical to the single-process trainer —
                # ft_load.py / 30_postcut / 32_qa_heldout need no changes. The
                # gathers are COLLECTIVES over the ORIGINAL modules (never the
                # shell — danger #14): every rank calls, rank 0 writes.
                _p_sd = _fsdp_full_sd(comp.projector)
                if IS_MAIN:
                    _asave(_p_sd, run_dir / "projector_latest.pt")
                _e_sd = None
                if ema_state is not None:
                    _e_sd = _fsdp_gather_dict(ema_state)  # sharded EMA -> full
                    if IS_MAIN:
                        _asave(_e_sd, run_dir / "projector_ema.pt")
                if lora_ps:
                    _l_sd = _fsdp_gather_dict(lora_state_dict(decoder))
                    if IS_MAIN:
                        _asave(_l_sd, run_dir / "lora_latest.pt")
                    if ema_lora is not None:
                        _le_sd = _fsdp_gather_dict(ema_lora)
                        if IS_MAIN:
                            _asave(_le_sd, run_dir / "lora_ema.pt")
                if enc_lora_ps:
                    _l_sd = _fsdp_gather_dict(lora_state_dict(comp.encoder))
                    if IS_MAIN:
                        _asave(_l_sd, run_dir / "enc_lora_latest.pt")
                    if ema_enc_lora is not None:
                        _le_sd = _fsdp_gather_dict(ema_enc_lora)
                        if IS_MAIN:
                            _asave(_le_sd, run_dir / "enc_lora_ema.pt")
                if args.full_ft and (step % args.full_save_every == 0 or step == args.steps):
                    # full_state_latest.pt keeps its name + is_full_state marker
                    # keys (enc/dec/projector/ema/step/args_hash) but LOSES
                    # opt/sched/py_rng — those live sharded in the DCP dir. The
                    # 'format' key makes any legacy resume attempt die loudly
                    # instead of half-restoring (danger #15).
                    _t_save = time.time()
                    _enc_sd = _fsdp_full_sd(comp.encoder)
                    _dec_sd = _fsdp_decoder_sd()
                    if IS_MAIN:
                        _asave({"enc": _enc_sd, "dec": _dec_sd, "projector": _p_sd,
                                "ema": _e_sd, "step": step, "args_hash": ARGS_HASH,
                                "format": "fsdp2-v1"}, state_path)
                        print(f"full-ft artifact state saved @ {step} "
                              f"({time.time() - _t_save:.0f}s)", flush=True)
                if (not args.full_ft) or step % args.full_save_every == 0 \
                        or step == args.steps:
                    # F9(a): async sharded RESUME checkpoint — each rank writes
                    # its own shard in parallel; committed (rename + LATEST
                    # pointer flip) once every rank's write lands (_dcp_poll).
                    if IS_MAIN:
                        heartbeat(step, note="saving")
                    _dcp_save(step)
            else:
                _asave(comp.projector.state_dict(), run_dir / "projector_latest.pt")
                if ema_state is not None:
                    _asave(ema_state, run_dir / "projector_ema.pt")
                if lora_ps:
                    _asave(lora_state_dict(decoder), run_dir / "lora_latest.pt")
                    if ema_lora is not None:
                        _asave(ema_lora, run_dir / "lora_ema.pt")
                if enc_lora_ps:
                    _asave(lora_state_dict(comp.encoder), run_dir / "enc_lora_latest.pt")
                    if ema_enc_lora is not None:
                        _asave(ema_enc_lora, run_dir / "enc_lora_ema.pt")
                if not args.full_ft:
                    _tmp = state_path.with_suffix(".tmp")  # atomic like the full-ft save (D8)
                    torch.save({"projector": comp.projector.state_dict(), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "py_rng": rng.getstate(), "step": step,
                                "ema": ema_state,
                                "lora": lora_state_dict(decoder) if lora_ps else None,
                                "ema_lora": ema_lora,
                                "enc_lora": lora_state_dict(comp.encoder) if enc_lora_ps else None,
                                "ema_enc_lora": ema_enc_lora,
                                "args_hash": ARGS_HASH},  # saved always, checked when present (r2 #25)
                               _tmp)
                    _tmp.rename(state_path)
                elif step % args.full_save_every == 0 or step == args.steps:
                    _t_save = time.time()
                    _tmp = state_path.with_suffix(".tmp")
                    torch.save({"enc": comp.encoder.state_dict(), "dec": decoder.state_dict(),
                                "projector": comp.projector.state_dict(), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "py_rng": rng.getstate(),
                                "step": step, "ema": ema_state,
                                "args_hash": ARGS_HASH}, _tmp)  # saved always (r2 #25)
                    _tmp.rename(state_path)  # atomic: crash mid-save keeps the previous file
                    print(f"full-ft resume state saved @ {step} ({time.time()-_t_save:.0f}s)", flush=True)
        if DIST:
            dist.barrier()  # ranks resume together only after rank 0's eval+save
            # the grad-ckpt gate rides module.training — a rank left in eval()
            # dies by OOM thousands of steps later (D8)
            assert not args.full_ft or (decoder.training and comp.training), \
                f"[rank {RANK}] train() not restored after eval block"
    if args.halt_after_step is not None and step >= args.halt_after_step:
        print(f"halt-after-step {args.halt_after_step}: stopping cleanly", flush=True)
        if IS_MAIN:
            # NOT done yet (validation): an async DCP save can still be
            # in flight — done rides the FINAL heartbeat after _dcp_wait +
            # the teardown barrier, so a hung final commit stays detectable.
            heartbeat(step, note="halt")
        _HALTED = True
        break
else:
    _HALTED = False

# halted runs skip the end-of-run exports: a projector_step{steps}/final_bf16
# stamped with the full step count would be a wrong-step showroom artifact
if FSDP_ON and not _HALTED:
    # F9(b): final exports gathered on ALL ranks (collectives), written by rank
    # 0 — formats and key sets unchanged, so ft_load/§4 evaluators just work
    if IS_MAIN:
        heartbeat(args.steps, note="saving")  # 75-min monitor allowance for exports
    _p_sd = _fsdp_full_sd(comp.projector)
    if IS_MAIN:
        torch.save(_p_sd, run_dir / f"projector_step{args.steps}.pt")
    if lora_ps:
        _l_sd = _fsdp_gather_dict(lora_state_dict(decoder))
        if IS_MAIN:
            torch.save(_l_sd, run_dir / f"lora_step{args.steps}.pt")
    if enc_lora_ps:
        _l_sd = _fsdp_gather_dict(lora_state_dict(comp.encoder))
        if IS_MAIN:
            torch.save(_l_sd, run_dir / f"enc_lora_step{args.steps}.pt")
    if args.full_ft:
        _enc_sd = _fsdp_full_sd(comp.encoder)
        _dec_sd = _fsdp_decoder_sd()
        if IS_MAIN:
            _asave({k: v.to(torch.bfloat16) for k, v in _enc_sd.items()},
                   run_dir / "encoder_final_bf16.pt")
            _asave({k: v.to(torch.bfloat16) for k, v in _dec_sd.items()},
                   run_dir / "decoder_final_bf16.pt")
elif IS_MAIN and not _HALTED:
    torch.save(comp.projector.state_dict(), run_dir / f"projector_step{args.steps}.pt")
    if lora_ps:
        torch.save(lora_state_dict(decoder), run_dir / f"lora_step{args.steps}.pt")
    if enc_lora_ps:
        torch.save(lora_state_dict(comp.encoder), run_dir / f"enc_lora_step{args.steps}.pt")
    if args.full_ft:
        # final weights in bf16 for eval/serving (halves disk; fp32 masters live in
        # full_state_latest.pt for any continuation)
        _asave({k: v.to(torch.bfloat16) for k, v in comp.encoder.state_dict().items()},
               run_dir / "encoder_final_bf16.pt")
        _asave({k: v.to(torch.bfloat16) for k, v in decoder.state_dict().items()},
               run_dir / "decoder_final_bf16.pt")
if DIST:
    if FSDP_ON:
        _dcp_wait()  # commit any in-flight async checkpoint before teardown (F9)
    dist.barrier()  # non-rank0 must not tear down the group under rank 0's exports
    dist.destroy_process_group()
if IS_MAIN:
    # completion is declared ONLY here — after every final artifact exists,
    # after the LAST async DCP checkpoint is committed (_dcp_wait) and after
    # the teardown barrier (validation + validation: done=True used to
    # precede _dcp_wait, so a hung/failed final DCP commit wore a permanently
    # 'done' heartbeat and the monitor stood down exactly when it was needed).
    heartbeat(step if _HALTED else args.steps,
              note="halt" if _HALTED else "complete", done=True)
print(f"done in {(time.time()-t0)/60:.1f} min — checkpoints in {run_dir}")
