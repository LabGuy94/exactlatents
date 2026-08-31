# Receipts

Raw per-example evidence (JSONL) behind every number reported in the docs. Each row is one
generation or one graded judgment. `pf` is the pooling factor: how many source tokens each
continuous latent vector replaces in the decoder context.

Provenance notes:

- The raw capture of the OOD-600 and in-dist-200 generation files carried a duplicated first
  row (601/201 lines, line 1 identical to line 2). The staged copies here are deduplicated to
  600/200 unique rows; nothing else was dropped.
- Bookkeeping fields (`_meta` argv/path entries, weight-file paths) were neutralized to the
  released repo layout. All graded content fields — `code`, `gen`, `question`, `answer`,
  `pred`, `reply` — are byte-verbatim from the evaluation runs. Some corpus functions are
  third-party code whose identifiers happen to name external products; those are corpus
  content and are left untouched.

| File | Lines | What it evidences |
|---|---|---|
| `ood600/ood600_pf4_gens.jsonl` | 600 | 600 OOD function generations at pf=4; backs 527/600 byte-exact, 577/600 code-exact |
| `ood600/ood600_pf6_gens.jsonl` | 600 | OOD-600 generations at pf=6; backs 522/600 code-exact |
| `ood600/ood600_pf8_gens.jsonl` | 600 | OOD-600 generations at pf=8; backs 389/600 code-exact |
| `ood600/ood600_pf12_gens.jsonl` | 600 | OOD-600 generations at pf=12; backs 6/600 code-exact |
| `ood600/ood600_text.jsonl` | 600 | Finetuned text-copy control on the same 600 functions; backs 565/600 byte-exact, 584/600 code-exact |
| `canaries/pf4_gens.jsonl` | 36 | Canary dev-set generations at pf=4; backs 30/36 byte-exact, 35/36 code-exact |
| `canaries/pf5_gens.jsonl` | 36 | Canary dev-set generations at pf=5 |
| `canaries/pf6_gens.jsonl` | 36 | Canary dev-set generations at pf=6 |
| `canaries/pf8_gens.jsonl` | 36 | Canary dev-set generations at pf=8 |
| `canaries/pf10_gens.jsonl` | 36 | Canary dev-set generations at pf=10 |
| `canaries/pf12_gens.jsonl` | 36 | Canary dev-set generations at pf=12 |
| `canaries/pf16_gens.jsonl` | 36 | Canary dev-set generations at pf=16 |
| `canaries/verified_gens.jsonl` | 36 | Canary generations from the final training run's verified checkpoint |
| `indist200/indist200_pf4_gens.jsonl` | 200 | 200 in-distribution (held-out training-corpus) generations at pf=4 |
| `indist200/indist200_text.jsonl` | 200 | Finetuned text-copy control on the same 200 in-distribution functions |
| `qa/qa_latents.jsonl` | 11399 | Held-out QA graded with latent-vector context at pf=4 (comprehension arm) |
| `qa/qa_latents_pf8.jsonl` | 3984 | Held-out QA with latent-vector context at pf=8 |
| `qa/qa_text_ft.jsonl` | 11399 | Held-out QA with full text context, finetuned decoder (parity baseline) |
| `qa/qa_text_stock.jsonl` | 11399 | Held-out QA with full text context, stock base model |
| `qa/qa_text_trunc25.jsonl` | 3984 | Held-out QA with text truncated to 25% (context-budget control) |
| `qa/qa_text_trunc125.jsonl` | 3984 | Held-out QA with text truncated to 12.5% (context-budget control) |
| `stacks/stacks_vec.jsonl` | 75 | Multi-function stacked-context generations from latent vectors |
| `stacks/stacks_text.jsonl` | 75 | Text-context control for the stacked-context setting |
| `retention/retention.jsonl` | 24 | 24 general-knowledge probes on the finetuned decoder; backs retention 19/24 |
| `controls/textcopy36.jsonl` | 36 | Text-copy control on the 36 canaries, finetuned decoder |
| `controls/textcopy36_bare.jsonl` | 36 | Text-copy control on the 36 canaries, bare prompt (no scaffold) |
| `controls/stock_ood600.jsonl` | 600 | Stock (non-finetuned) base model text-copy on OOD-600 |
| `controls/stock_indist200.jsonl` | 200 | Stock base model text-copy on the in-distribution 200 |
| `controls/stock_stacks.jsonl` | 75 | Stock base model control for the stacked-context setting |
| `controls/stock_bare36.jsonl` | 36 | Stock base model bare-prompt control on the 36 canaries |
| `ood_qa/questions_frozen.jsonl.gz` | 1268 | Frozen fully-OOD QA question set (questions generated before any arm answered) |
| `ood_qa/vec.jsonl` | 1268 | Fully-OOD QA graded answers, latent-vector arm; backs 39.0% |
| `ood_qa/ft_text.jsonl` | 1268 | Fully-OOD QA graded answers, finetuned text arm; backs 39.6% (parity) |
| `ood_qa/stock_text.jsonl` | 1268 | Fully-OOD QA graded answers, stock base model text arm |
