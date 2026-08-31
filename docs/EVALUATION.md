# Evaluation

This document defines the release metrics, battery arms, canonical corrections, and corpus provenance. Per-item JSONL receipts are the evidence; the scripts under [`eval/`](../eval/) re-derive the tables in [RESULTS.md](RESULTS.md).

## Exactness metrics

Both metrics are implemented in [`compressor/exactness.py`](../compressor/exactness.py).

### Byte-exact

`byte_exact(reference, generation)` requires the generation to begin with the complete reference byte for byte. The next character must be a newline, the generation must end at that boundary, or the reference must already end in a newline. There is no whitespace stripping or fuzzy prefix allowance.

### Code-exact

`code_exact(reference, generation)` compares normalized, parseable Python:

- AST docstring constants are replaced with neutral placeholders.
- Tokenizer-recognized comments are removed.
- Trailing whitespace is removed and blank lines are collapsed.
- Indentation, executable string literals, identifiers, punctuation, and other code-bearing text remain significant.
- Plausible newline cut points in the free-running generation are tried so continuation after a complete function does not create a false miss.
- If the reference or candidate does not parse, that candidate is not code-exact.

Code-exact is therefore prose-blind exact source comparison, not semantic equivalence and not arbitrary AST equivalence.

## What the pooling factor measures

For a source of token length `T` and pooling factor `p`, the decoder receives approximately [`ceil(T / p) + 2` latent positions](../eval/ood_battery.py), including learned boundaries. The nominal [PF4 result](../receipts/ood600/ood600_pf4_gens.jsonl) averages [3.73× fewer decoder context slots](../receipts/ood600/ood600_pf4_gens.jsonl) on the OOD battery.

A latent position is a full dense floating-point vector, not a token identifier. The release therefore claims fewer decoder context slots and KV-cache positions, not storage or bit-rate compression.

## Reconstruction batteries

All published reconstruction generations are [single-sample greedy decodes (`do_sample=False`; temperature zero)](../eval/ood_battery.py). The arms are:

- **OOD:** compressed vectors at [PF4, PF6, PF8, and PF12](../eval/ood_battery.py), plus labeled full-text copying by the released and stock decoders.
- **In-distribution:** compressed vectors at [PF4](../receipts/indist200/indist200_pf4_gens.jsonl), plus the same full-text controls.
- **Canary development set:** compressed vectors at [PF4, PF5, PF6, PF8, PF10, PF12, and PF16](../eval/canaries.py), with labeled and bare-text controls. This is a development set because it was inspected during model selection.
- **Stacks:** a solo target or a labeled context of [20 same-repository or mixed-repository functions](../eval/stacks.py), evaluated with latent and full-text contexts.

Every arm uses the same exactness definitions. Text-copy controls repeat a function label around the source; bare-text controls intentionally omit a copy instruction and measure default continuation under that framing.

### In-distribution parse caveat

The in-distribution sample contains [15/200 references that do not parse standalone](../receipts/indist200/indist200_pf4_gens.jsonl), largely because a method body is stored with class-level indentation or decoration. [Fourteen latent outputs](../receipts/indist200/indist200_pf4_gens.jsonl) reproduce such references byte-exactly but cannot pass code-exact because the reference itself is unparseable. The release reports the conservative metric without silently filtering those rows.

### Stack formatting caveat

The stock stack decoder often wraps otherwise usable code in Markdown fences. The canonical stock stack result strips only a leading and matching trailing fence before applying the same exactness scorer, recovering [27/150 byte-exact and 29/150 code-exact outputs](../receipts/controls/stock_stacks.jsonl) across the stored stacked conditions. The released latent and text arms are scored without that stock-only presentation repair; their raw outputs do not exhibit the same systematic fencing. Stock solo generations were not retained, so no corrected solo stock count is reported.

## QA protocol

The standard QA battery presents either compressed vectors or raw code, followed by the identical `Q:` / `A:` prompt. It covers return-value dataflow and execution/exception intents. Those question types were withheld from QA training, but their functions were present as reconstruction-training examples; this is question-type generalization, not function-level generalization.

The standard harness uses greedy decoding with a gold-conditioned limit of [`max(16, min(gold_tokens + 8, 64))` new tokens](../eval/qa.py). Full-text prefix controls retain [25%](../receipts/qa/qa_text_trunc25.jsonl) or [12.5%](../receipts/qa/qa_text_trunc125.jsonl) of source tokens. These are budget-oriented truncation controls, not exact position matches: vector contexts include boundaries and ceiling-rounded pools.

The fully OOD QA battery freezes [1,268 questions](../receipts/ood_qa/questions_frozen.jsonl.gz) over OOD functions and runs vectors, released full text, and stock full text in that order. Every arm uses [greedy decoding with at most 64 new tokens](../eval/ood_qa/). Its functions and question types are both out of distribution relative to reconstruction and QA training.

QA answers use the shared type-aware grader in the public dataset package. The reported comparisons are paired because every arm receives the same frozen question records.

## QA extraction and grading caveats

### First-line and length truncation

The historical extractor took only the first generated line, stripped it, and capped it at [200 characters](../eval/qa.py) before grading. This can erase a multiline answer or turn a generation beginning with a newline into an empty prediction. Long `name` and `substring` answers can also be structurally impossible to pass under the character and generation caps.

The held-out comparison is minimally affected: [one of 3,983 gold answers](../receipts/qa/qa_latents.jsonl) exceeds the character cap, fails in every arm, and changes corrected rates by no more than [0.02 percentage point](../receipts/qa/qa_latents.jsonl). The caveat matters more for long verbatim answers and should not be generalized away from this battery.

### Glued-template continuation

The full-text decoder sometimes emits the correct answer and immediately continues with `Q:` or `A:` without a newline. A first-line split cannot detect this. Release analysis therefore also cuts at the first literal prompt marker. [`eval/qa.py`](../eval/qa.py) intentionally preserves the historical first-line grader; the correction is applied when deriving the published table from its raw receipts.

In the standard held-out full-text arm, [132 predictions](../receipts/qa/qa_text_ft.jsonl) were unambiguous exact-gold answers followed by a glued template. Correcting only those raises the canonical full-text count from [2,018/3,983](../receipts/qa/qa_text_ft.jsonl) to [2,150/3,983](../receipts/qa/qa_text_ft.jsonl). The latent count remains [2,087/3,983](../receipts/qa/qa_latents.jsonl). The resulting paired comparison is [statistically indistinguishable, exact two-sided p=0.050](../receipts/qa/qa_text_ft.jsonl), with the nominal direction favoring text.

The fully OOD battery applies the same rule. It leaves vectors at [494/1,268](../receipts/ood_qa/vec.jsonl) and raises full text from [483/1,268](../receipts/ood_qa/ft_text.jsonl) to [502/1,268](../receipts/ood_qa/ft_text.jsonl). These termination-aware values are canonical because they score the answer before an arm-specific stopping failure rather than treating template continuation as a comprehension error.

### Grader asymmetries

Single-name answers pass by identifier-boundary containment, while multi-name answers require the expected ordered list. The maximum generation length is derived from the gold answer length, which gives every arm the same answer-length hint. These choices are arm-symmetric but make absolute accuracy grader-specific.

## Canonical control rescoring

An earlier text-copy control harness passed `generation, reference` into functions whose public signature is `reference, generation`. Reversing the arguments particularly penalizes a generation that is a correct copy followed by a continuation. Vector-side harnesses used the correct order.

The release rescored stored generations with [`eval/rescore.py`](../eval/rescore.py), which calls `byte_exact(reference, generation)` and `code_exact(reference, generation)`. The canonical corrected controls are:

| Control | Byte-exact | Code-exact |
|---|---:|---:|
| Released decoder, OOD full text | [565/600](../receipts/ood600/ood600_text.jsonl) | [584/600](../receipts/ood600/ood600_text.jsonl) |
| Stock decoder, OOD full text | [496/600](../receipts/controls/stock_ood600.jsonl) | [510/600](../receipts/controls/stock_ood600.jsonl) |
| Released decoder, in-distribution full text | [185/200](../receipts/indist200/indist200_text.jsonl) | [175/200](../receipts/indist200/indist200_text.jsonl) |
| Stock decoder, in-distribution full text | [175/200](../receipts/controls/stock_indist200.jsonl) | [164/200](../receipts/controls/stock_indist200.jsonl) |

These values, not summary flags embedded by the old harness, are canonical because they apply the same metric direction used by every vector receipt.

## Receipt deduplication

The OOD generation files are append-safe and carry a duplicated first successful row after resume. [`eval/rescore.py`](../eval/rescore.py) deduplicates by stable sample identity before scoring, yielding the canonical [600 distinct functions](../eval/rescore.py) for [PF4](../receipts/ood600/ood600_pf4_gens.jsonl), [PF6](../receipts/ood600/ood600_pf6_gens.jsonl), [PF8](../receipts/ood600/ood600_pf8_gens.jsonl), and [PF12](../receipts/ood600/ood600_pf12_gens.jsonl). Scoring raw line count would double-count that sample.

The in-distribution latent receipt has the same append/resume duplicate; [`eval/rescore.py`](../eval/rescore.py) deduplicates it to [200 stable sample identities](../receipts/indist200/indist200_pf4_gens.jsonl) before scoring.

## OOD corpus construction and contamination accounting

The OOD pool was built from pinned commits in [199 repositories created after the base model's knowledge cutoff](../eval/ood_battery.py). Candidate functions were filtered by length and parseability, then exact-hash deduplicated against the complete reconstruction corpus before sampling balanced length bands.

A full audit found [no exact training-corpus match](../eval/ood_battery.py). It found [one whitespace-normalized overlap](../eval/ood_battery.py); that function was [not byte-exact in the OOD evaluation](../receipts/ood600/ood600_pf4_gens.jsonl). Consequently, [zero successful reconstructions receive contamination credit](../receipts/ood600/ood600_pf4_gens.jsonl).

This construction supports function-level out-of-distribution reconstruction and the fully OOD QA battery. It does not establish cross-language, cross-model, or multi-seed generalization.
