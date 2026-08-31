#!/usr/bin/env bash
# Exact recipe used for the released checkpoint's final training run.
# Hardware: 8x H100; about 20,000 steps; checkpoint at 16000.
# Build the corpus with dataset/prepare.py and the corpus tools in dataset/.
# Generate the QA sidecar with the pregeneration tools in dataset/ before launch.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MIX="recon=0.45,qapre=0.15,plain=0.10,span=0.08,stack=0.08,desc=0.05,retr=0.05,cont=0.04"

CARRY=(
  --pf 4
  --pf-mix 4=0.7,8=0.3
  --pooling latent
  --boundary
  --eos
  --task-markers
  --distractor-frac 0.5
  --stack-n-max 40
  --stack-token-budget 12000
  --rank-loss 0.5
  --rank-margin 3
  --rank-ramp 1000
  --mask-p 0.5
  --prose-weight 0.3
  --lr-decay
  --ema 0.999
  --chunked-loss
  --loss-chunk 1024
  --surprisal-weight 0
)

COMMON=(
  --distributed
  --fsdp
  --full-ft
  --dec-grad-ckpt
  --enc-grad-ckpt
  --hf-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
  --data data/corpus_v2/corpus
  --qa-data data/corpus_v2/qa_phrased
  --long-buckets
  --token-budget 6144
  --qapre-conc-frac 0.2
  --mix "$MIX"
  "${CARRY[@]}"
  --eval-every 250
  --gen-every 2000
  --full-save-every 2000
  --gen-canaries 20
  --gen-batch 20
)

PERFORMANCE=(
  --telemetry-batched
  --fsdp-accum-hold
  --build-prefetch
  --ckpt-token-threshold 6144
  --pad-to-max
)

TRAIN=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node=8
  train/train_reconstruction.py
  "${COMMON[@]}"
  --steps 20000
  --batch 4
  --grad-accum 8
  --eff-batch 256
  "${PERFORMANCE[@]}"
  --run-name train
)

case "${1:-train}" in
  train)
    "${TRAIN[@]}"
    ;;
  resume)
    "${TRAIN[@]}" --resume
    ;;
  *)
    printf 'usage: %s [train|resume]\n' "$0" >&2
    exit 2
    ;;
esac
