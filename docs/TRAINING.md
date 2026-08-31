# Training

ExactLatents is a full fine-tune of an encoder, a learned latent projector, and a decoder. It is trained end to end so that the decoder can consume continuous vectors through `inputs_embeds`, reconstruct the original Python function, and answer questions from the same vector representation.

## Architecture

The encoder and decoder both start from Qwen3-1.7B. Source tokens pass through the encoder, a learned attention pooler allocates a shorter sequence of 2,048-dimensional latent slots, and a two-layer MLP maps those slots into the decoder embedding space. Learned vectors mark the beginning and end of each latent block.

| Component | Configuration | Size |
|---|---|---:|
| Encoder | Qwen3-1.7B trunk with the final two transformer blocks removed (26 of 28 blocks retained) | 1.62B unique trainable parameters |
| Latent projector | Learned 16-head attention pooler, two-layer 2,048 → 2,048 → 2,048 MLP, and boundary vectors | 0.03B trainable parameters |
| Decoder | Complete Qwen3-1.7B decoder | 2.03B serialized tensor elements; 1.72B unique trainable parameters after tied weights are counted once |
| Total | Encoder, projector, and decoder trained together | 3.37B unique trainable parameters |

The distinction in the decoder row matters: the serialized state counts tied token-embedding and output-head storage in its tensor inventory, while the trainable total counts that shared parameter once.

At pooling factor `pf`, the pooler emits approximately one latent slot per `pf` source tokens. The released recipe trains one shared model at `pf=4` and `pf=8`; it does not train separate projectors for the two rates.

## Training data

The corpus contains 2.6 million Python functions. The preparation pipeline under [`dataset/`](../dataset/) extracts functions, deduplicates them, applies repository and quality filters, creates the deterministic training draw, and prepares the question-answer sidecar used by the multitask trainer.

Each micro-batch independently draws its pooling factor:

- `pf=4`: 70%
- `pf=8`: 30%

Comments and docstrings receive `0.3×` the token loss of executable code. This asks the model to preserve prose without allowing it to dominate exact reconstruction of identifiers, literals, and syntax.

## Task mixture

One task arm is sampled per micro-batch. The weights sum to 1.00.

| Arm | Weight | Objective |
|---|---:|---|
| Reconstruction (`recon`) | 0.45 | Reconstruct the exact original function from its latent vectors. |
| QA from vectors (`qapre`) | 0.15 | Answer a pregenerated question using a solo or stacked latent context. One fifth of this arm is reserved for conceptual questions. |
| Plain LM rehearsal (`plain`) | 0.10 | Apply the base next-token objective to raw code, without vectors, to reduce forgetting. |
| Span QA (`span`) | 0.08 | Recover a requested substring or local fact from a latent context. |
| Stack targeting (`stack`) | 0.08 | Select a named function from a labeled stack of latent blocks and reproduce it. |
| Description (`desc`) | 0.05 | Predict a function's docstring from prose-stripped code vectors. |
| Retrieval (`retr`) | 0.05 | Answer a retrieval question over a shared stack of latent functions. |
| Continuation (`cont`) | 0.04 | Encode the first half of a function and predict the second half. |

All QA-bearing arms use latent context. The plain arm is ordinary raw-code language-model rehearsal, not text-context QA.

## Optimization recipe

The final recipe is a single-node `8×H100` full fine-tune with full parameter sharding and gradient checkpointing on both backbones.

| Setting | Value |
|---|---:|
| Planned optimizer steps | 20,000 |
| Released checkpoint | step 16,000 |
| Batch geometry | 8 ranks × 4 examples per rank × 8 accumulation micro-batches; nominal effective batch 256 |
| Per-rank source-token budget | 6,144 |
| Projector and pooler learning rate | `5e-4` |
| Encoder learning rate | `1e-5` |
| Decoder learning rate | `1e-5` |
| Schedule | 100-step linear warmup, then cosine decay to 5% of peak |
| Precision | fp32 master parameters and optimizer state; bf16 autocast compute |
| Projector EMA | 0.999 decay |

The reconstruction objective also uses learned block boundaries, an explicit end-of-sequence target, task markers, closed-book masking of half the selected rare-token history, and a non-prose ranking loss. Half of reconstruction batches include an unrelated prefix so the decoder learns to read latent blocks in context rather than only at position zero.

[`train/launch.sh`](../train/launch.sh) is the exact launch invocation, including the full task mix, pooling-factor distribution, optimizer geometry, loss settings, and checkpoint cadence. Use the scripts under [`dataset/`](../dataset/) to construct the corpus and QA inputs expected by that command.