# Limitations

ExactLatents demonstrates that Python functions can be represented in fewer decoder context positions while retaining high reconstruction fidelity and useful question answering. It does not establish a general-purpose or storage-efficient codec.

## Context slots are not stored bits

The pooling factor measures decoder positions: a source with `T` tokens becomes approximately `ceil(T / pf) + 2` latent positions, including two learned boundary positions. At `pf=4`, the measured mean source-token-to-latent-position ratio is 3.83× on the canary development set and 3.73× on the 600-function out-of-distribution set. The accurate claim is therefore **roughly 4× fewer decoder context slots**, not 4× storage compression.

A latent position is also much larger than a token identifier. One 2,048-dimensional bf16 vector occupies about 32,768 bits, while a token identifier carries roughly 17 bits. Considered only as stored data, the latent representation is approximately a 500× bit expansion. Its benefit is reduced decoder sequence length and KV-cache positions, not reduced bytes on disk or on the wire.

## Evaluation scope

The evidence is deliberately narrow:

- one training seed;
- one programming language, Python;
- one base-model choice, Qwen3-1.7B;
- greedy reconstruction under the released evaluation protocol.

The results do not establish the same behavior for other seeds, model scales, architectures, natural-language corpora, or programming languages. Multi-seed confidence intervals and cross-model replication remain open.

## Question-answering caveats

One intent-held-out QA evaluation uses functions that were present in the reconstruction training corpus. The question types were held out, but the functions themselves were not, so that evaluation cannot establish function-level generalization. Its full-text comparison is also made with a model whose QA training used vector context rather than text context.

A separate fully out-of-distribution QA battery addresses the function-overlap caveat: accuracy was 39.0% from latent context and 39.6% from full-text context. These results are **statistically indistinguishable**. They support parity on that battery, not a claim that latents beat text.

## Reconstruction is not guaranteed

Byte-exact reconstruction remains probabilistic model behavior, not a lossless codec guarantee. On the 600-function out-of-distribution set, code-exact reconstruction declines as fewer slots are used: 577/600 at `pf=4`, 522/600 at `pf=6`, 389/600 at `pf=8`, and 6/600 at `pf=12`. Only `pf=4` and `pf=8` were represented in training.

Length also matters. At `pf=4`, code-exact results by source length were 119/120 for at most 64 tokens, 118/120 for 65–128 tokens, 116/120 for 129–256 tokens, and 224/240 above 256 tokens. Longer inputs leave more opportunities for an early autoregressive error to cascade, so users should not extrapolate short-function accuracy to arbitrarily long code.

## Retention cost

Full fine-tuning changes the decoder outside the latent-reading task. On a small 24-item general-capability spot check, the released model scored 19/24 while the untouched base model scored 24/24. This is a measured retention cost, and the small benchmark is not broad enough to characterize every capability that may have changed.

## Related-work boundary

The distinguishing evidence here is the combination of **per-sample byte-exact reconstruction from continuous vectors** and **useful queries over that same representation**. The boundary is narrower than “first lossless compression” or “first learned context compression.”

- **[CCF (arXiv:2509.09199)](https://arxiv.org/abs/2509.09199):** aggregate ROUGE-L of 1.00 does not show that each program is byte-identical; this release scores per-sample bytes and queries the same continuous vectors.
- **[Tang et al., Compression Is Routing (arXiv:2512.16963)](https://arxiv.org/abs/2512.16963):** reports token-level reconstruction with a small autoencoder and specialist repeater, while this release measures byte-exact programs and uses a general decoder for both replay and QA.
- **[C3 (arXiv:2511.15244)](https://arxiv.org/abs/2511.15244):** evaluates character precision on OCR-oriented prose with a reconstruction-specialized reader; this release measures program bytes per sample and demonstrates queries over the replayable vectors.
- **[The Lossy Horizon (arXiv:2510.22207)](https://arxiv.org/abs/2510.22207):** obtains exact recovery through residual side information after learned reconstruction; this release's byte-exact metric is measured from the vectors themselves, which remain queryable.
- **Gist Tokens:** optimize compact soft prompts for preserving task behavior rather than recovering the source byte for byte; this release tests exact replay and QA from one representation.
- **ICAE:** learns memory slots for long-context reconstruction and continuation, but does not establish this release's paired result of per-program byte exactness and querying over the same released code vectors.

These distinctions do not imply that the underlying mechanisms are wholly new. They state the specific empirical combination measured by this release.