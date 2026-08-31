# Results

These are the release's canonical results. Exactness means the definitions in [EVALUATION.md](EVALUATION.md), and every count below points to a released per-item receipt except the explicitly marked stock-retention summary. The scope is deliberately narrow: a [single training seed](../eval/ood_battery.py), [Python only](../eval/ood_battery.py), and one base-model family, [Qwen3-1.7B](../eval/ood_battery.py). A nominal pooling factor describes decoder context slots, not stored bits.

## Out-of-distribution reconstruction

The primary result is [527/600 byte-exact](../receipts/ood600/ood600_pf4_gens.jsonl) and [577/600 code-exact](../receipts/ood600/ood600_pf4_gens.jsonl) at [PF4](../receipts/ood600/ood600_pf4_gens.jsonl). Including the learned boundary slots, this uses a mean [3.73× fewer decoder context slots](../receipts/ood600/ood600_pf4_gens.jsonl) than the tokenized source. It is not a bit-rate compression claim.

| Pooling factor | Byte-exact | Code-exact | Mean decoder-slot ratio |
|---:|---:|---:|---:|
| [PF4](../receipts/ood600/ood600_pf4_gens.jsonl) | [527/600 (87.83%)](../receipts/ood600/ood600_pf4_gens.jsonl) | [577/600 (96.17%)](../receipts/ood600/ood600_pf4_gens.jsonl) | [3.73×](../receipts/ood600/ood600_pf4_gens.jsonl) |
| [PF6](../receipts/ood600/ood600_pf6_gens.jsonl) | [418/600 (69.67%)](../receipts/ood600/ood600_pf6_gens.jsonl) | [522/600 (87.00%)](../receipts/ood600/ood600_pf6_gens.jsonl) | — |
| [PF8](../receipts/ood600/ood600_pf8_gens.jsonl) | [264/600 (44.00%)](../receipts/ood600/ood600_pf8_gens.jsonl) | [389/600 (64.83%)](../receipts/ood600/ood600_pf8_gens.jsonl) | — |
| [PF12](../receipts/ood600/ood600_pf12_gens.jsonl) | [1/600 (0.17%)](../receipts/ood600/ood600_pf12_gens.jsonl) | [6/600 (1.00%)](../receipts/ood600/ood600_pf12_gens.jsonl) | — |

At [PF4](../receipts/ood600/ood600_pf4_gens.jsonl), code exactness degrades gradually with source length rather than showing a short-function-only result:

| Source-token band | Code-exact |
|---:|---:|
| [≤64](../receipts/ood600/ood600_pf4_gens.jsonl) | [119/120 (99.17%)](../receipts/ood600/ood600_pf4_gens.jsonl) |
| [65–128](../receipts/ood600/ood600_pf4_gens.jsonl) | [118/120 (98.33%)](../receipts/ood600/ood600_pf4_gens.jsonl) |
| [129–256](../receipts/ood600/ood600_pf4_gens.jsonl) | [116/120 (96.67%)](../receipts/ood600/ood600_pf4_gens.jsonl) |
| [>256](../receipts/ood600/ood600_pf4_gens.jsonl) | [224/240 (93.33%)](../receipts/ood600/ood600_pf4_gens.jsonl) |

**Re-derive:** Re-grade the linked receipts with [`eval/rescore.py`](../eval/rescore.py); regenerate each rate with [`eval/ood_battery.py --pf N`](../eval/ood_battery.py). For the length table, group the regenerated [PF4](../receipts/ood600/ood600_pf4_gens.jsonl) result rows by `n_tok` at the listed cut points and sum `code_exact`.

## Canary development sweep

This is a repeatedly inspected development set, not the primary held-out result.

| Pooling factor | Byte-exact | Code-exact |
|---:|---:|---:|
| [PF4](../receipts/canaries/pf4_gens.jsonl) | [30/36 (83.33%)](../receipts/canaries/pf4_gens.jsonl) | [35/36 (97.22%)](../receipts/canaries/pf4_gens.jsonl) |
| [PF5](../receipts/canaries/pf5_gens.jsonl) | [26/36 (72.22%)](../receipts/canaries/pf5_gens.jsonl) | [35/36 (97.22%)](../receipts/canaries/pf5_gens.jsonl) |
| [PF6](../receipts/canaries/pf6_gens.jsonl) | [19/36 (52.78%)](../receipts/canaries/pf6_gens.jsonl) | [31/36 (86.11%)](../receipts/canaries/pf6_gens.jsonl) |
| [PF8](../receipts/canaries/pf8_gens.jsonl) | [6/36 (16.67%)](../receipts/canaries/pf8_gens.jsonl) | [24/36 (66.67%)](../receipts/canaries/pf8_gens.jsonl) |
| [PF10](../receipts/canaries/pf10_gens.jsonl) | [1/36 (2.78%)](../receipts/canaries/pf10_gens.jsonl) | [7/36 (19.44%)](../receipts/canaries/pf10_gens.jsonl) |
| [PF12](../receipts/canaries/pf12_gens.jsonl) | [0/36](../receipts/canaries/pf12_gens.jsonl) | [0/36](../receipts/canaries/pf12_gens.jsonl) |
| [PF16](../receipts/canaries/pf16_gens.jsonl) | [0/36](../receipts/canaries/pf16_gens.jsonl) | [0/36](../receipts/canaries/pf16_gens.jsonl) |

Only [PF4 and PF8](../eval/canaries.py) were represented in training; the other rates test interpolation or extrapolation. The observed collapse above the trained range is therefore a property of this checkpoint, not an information-theoretic limit.

**Re-derive:** Re-grade the linked receipts with [`eval/rescore.py`](../eval/rescore.py); regenerate the sweep with [`eval/canaries.py`](../eval/canaries.py).

## Direct-text and stock-decoder controls

All byte counts here use the canonical scorer argument order described in [EVALUATION.md](EVALUATION.md). Earlier summaries produced by the control harness are not authoritative.

| Set and arm | Byte-exact | Code-exact |
|---|---:|---:|
| OOD, released decoder copying labeled full text | [565/600 (94.17%)](../receipts/ood600/ood600_text.jsonl) | [584/600 (97.33%)](../receipts/ood600/ood600_text.jsonl) |
| OOD, stock decoder copying labeled full text | [496/600 (82.67%)](../receipts/controls/stock_ood600.jsonl) | [510/600 (85.00%)](../receipts/controls/stock_ood600.jsonl) |
| In-distribution, released decoder copying labeled full text | [185/200 (92.50%)](../receipts/indist200/indist200_text.jsonl) | [175/200 (87.50%)](../receipts/indist200/indist200_text.jsonl) |
| In-distribution, stock decoder copying labeled full text | [175/200 (87.50%)](../receipts/controls/stock_indist200.jsonl) | [164/200 (82.00%)](../receipts/controls/stock_indist200.jsonl) |
| Canary development set, released decoder copying labeled full text | [34/36 (94.44%)](../receipts/controls/textcopy36.jsonl) | [34/36 (94.44%)](../receipts/controls/textcopy36.jsonl) |
| Canary development set, released decoder given bare text | [0/36](../receipts/controls/textcopy36_bare.jsonl) | [0/36](../receipts/controls/textcopy36_bare.jsonl) |
| Canary development set, stock decoder given bare text | [0/36](../receipts/controls/stock_bare36.jsonl) | [0/36](../receipts/controls/stock_bare36.jsonl) |

The bare-text cells measure default continuation behavior under that exact framing; they do not show that either decoder is unable to copy when explicitly asked.

**Re-derive:** Re-grade each linked text-control receipt with [`eval/rescore.py`](../eval/rescore.py), which calls the public exactness functions in canonical argument order.

## In-distribution reconstruction

| Arm | Byte-exact | Code-exact |
|---|---:|---:|
| Latents at [PF4](../receipts/indist200/indist200_pf4_gens.jsonl) | [184/200 (92.00%)](../receipts/indist200/indist200_pf4_gens.jsonl) | [178/200 (89.00%)](../receipts/indist200/indist200_pf4_gens.jsonl) |
| Released decoder, labeled full text | [185/200 (92.50%)](../receipts/indist200/indist200_text.jsonl) | [175/200 (87.50%)](../receipts/indist200/indist200_text.jsonl) |
| Stock decoder, labeled full text | [175/200 (87.50%)](../receipts/controls/stock_indist200.jsonl) | [164/200 (82.00%)](../receipts/controls/stock_indist200.jsonl) |

[Fourteen byte-exact rows](../receipts/indist200/indist200_pf4_gens.jsonl) are not code-exact because their stored references do not parse as standalone Python; [15/200 references](../receipts/indist200/indist200_pf4_gens.jsonl) have this property. The metric conservatively treats an unparseable reference as a code-exact miss.

**Re-derive:** Re-grade the linked receipts with [`eval/rescore.py`](../eval/rescore.py); duplicate sample identities are removed before scoring.

## Multi-function stacks

Each stack condition asks the decoder to reproduce one target. Stacked conditions provide [20 functions](../eval/stacks.py) from either one repository or mixed repositories.

| Representation | Solo byte / code | Same-repository byte / code | Mixed-repository byte / code |
|---|---:|---:|---:|
| Latents | [72/75 / 75/75](../receipts/stacks/stacks_vec.jsonl) | [44/75 / 46/75](../receipts/stacks/stacks_vec.jsonl) | [53/75 / 56/75](../receipts/stacks/stacks_vec.jsonl) |
| Labeled full text, released decoder | [74/75 / 74/75](../receipts/stacks/stacks_text.jsonl) | [1/75 / 52/75](../receipts/stacks/stacks_text.jsonl) | [1/75 / 57/75](../receipts/stacks/stacks_text.jsonl) |
| Labeled full text, stock decoder | — | [27/150 byte / 29/150 code across the two stacked conditions](../receipts/controls/stock_stacks.jsonl) | — |

In the mixed-repository condition, [all 19 latent failures](../receipts/stacks/stacks_vec.jsonl) and [17 of 18 full-text failures](../receipts/stacks/stacks_text.jsonl) target the repeated name `__init__`. This concentration in both representations points to retrieval-key collision rather than a vector-specific failure. The stock count strips leading Markdown fences before exactness scoring; its solo generations were not retained.

**Re-derive:** [`eval/stacks.py`](../eval/stacks.py).

## Question answering on training-corpus functions

These functions were seen as reconstruction targets during training. The held-out axis is the question type: return-value dataflow and execution/exception questions were withheld from QA training. The table uses termination-aware extraction, which cuts glued prompt-template continuations as described in [EVALUATION.md](EVALUATION.md).

| Arm | Correct | Accuracy |
|---|---:|---:|
| Latents at [PF4](../receipts/qa/qa_latents.jsonl) | [2,087/3,983](../receipts/qa/qa_latents.jsonl) | [52.40%](../receipts/qa/qa_latents.jsonl) |
| Released decoder, full text | [2,150/3,983](../receipts/qa/qa_text_ft.jsonl) | [53.98%](../receipts/qa/qa_text_ft.jsonl) |
| Latents at [PF8](../receipts/qa/qa_latents_pf8.jsonl) | [1,849/3,983](../receipts/qa/qa_latents_pf8.jsonl) | [46.42%](../receipts/qa/qa_latents_pf8.jsonl) |
| Released decoder, text prefix truncated to [25%](../receipts/qa/qa_text_trunc25.jsonl) | [1,745/3,983](../receipts/qa/qa_text_trunc25.jsonl) | [43.81%](../receipts/qa/qa_text_trunc25.jsonl) |
| Released decoder, text prefix truncated to [12.5%](../receipts/qa/qa_text_trunc125.jsonl) | [1,599/3,983](../receipts/qa/qa_text_trunc125.jsonl) | [40.15%](../receipts/qa/qa_text_trunc125.jsonl) |
| Stock decoder, full text | [1,137/3,983](../receipts/qa/qa_text_stock.jsonl) | [28.55%](../receipts/qa/qa_text_stock.jsonl) |

Latents and untruncated text are [statistically indistinguishable under the paired test, exact two-sided p=0.050](../receipts/qa/qa_text_ft.jsonl). The nominal direction is text-favoring after correction. The text arm is not a best-achievable text-QA baseline: the released decoder's QA training used vector contexts, not text contexts. Prefix truncation is a budget-oriented control, but short examples and boundary slots prevent exact position equality.

**Re-derive:** [`eval/qa.py`](../eval/qa.py) preserves the historical first-line grader. Apply the termination correction in [EVALUATION.md](EVALUATION.md) to the linked raw receipts to obtain the canonical table.

## Fully out-of-distribution question answering

This battery applies held-out question types to the OOD functions, removing the function-seen-during-reconstruction caveat above.

| Arm | Correct | Accuracy |
|---|---:|---:|
| Latents at [PF4](../receipts/ood_qa/vec.jsonl) | [494/1,268](../receipts/ood_qa/vec.jsonl) | [38.96% (39.0%)](../receipts/ood_qa/vec.jsonl) |
| Released decoder, full text | [502/1,268](../receipts/ood_qa/ft_text.jsonl) | [39.59% (39.6%)](../receipts/ood_qa/ft_text.jsonl) |
| Stock decoder, full text | [214/1,268](../receipts/ood_qa/stock_text.jsonl) | [16.88%](../receipts/ood_qa/stock_text.jsonl) |

Latents and untruncated text are [statistically indistinguishable, exact two-sided paired p=0.686](../receipts/ood_qa/ft_text.jsonl). This is parity, not evidence that latents beat text.

**Re-derive:** [`eval/ood_qa/generate_ood_qa.py`](../eval/ood_qa/generate_ood_qa.py) records both extractions, and [`eval/ood_qa/grade_ood_qa.py`](../eval/ood_qa/grade_ood_qa.py) verifies the receipts and paired comparison.

## General-capability retention

| Decoder | Correct |
|---|---:|
| Released checkpoint | [19/24](../receipts/retention/retention.jsonl) |
| Untouched stock model | 24/24 (stock-decoder control run; no per-item stock receipt is included) |

This small spot-check records a retention cost from full finetuning; it is not a broad benchmark.

**Re-derive:** Sum the per-item `ok` flags in the released-checkpoint [retention receipt](../receipts/retention/retention.jsonl). The stock comparison is retained as a control-run summary because its per-item receipt is unavailable.
