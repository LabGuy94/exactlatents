# ExactLatents playground

`playground.py` is an interactive terminal for reconstructing Python functions
from continuous latent vectors. It compares three inputs in one session:

- **vec** — the released finetuned decoder reads vectors produced by the
  encoder, latent pooler, and projector.
- **text** — the same finetuned decoder reads the original code tokens.
- **stock** — an untouched Qwen3-1.7B model reads the original code tokens.

At pooling factor 4, the vector arm uses about four times fewer decoder context
slots than raw token text. This is a context-length reduction, not a reduction
in stored bits: one bf16 vector is roughly a 500-times bit expansion over one
token ID.

## Quickstart

From the repository root:

```console
$ hf download labguy/exactlatents-qwen3-1.7b --local-dir weights/
$ uv run python playground.py
```

Then type `demo` for the guided tour, or `paste` to try your own function.
The two required weight files are:

```text
weights/model.safetensors
weights/projector_ema.safetensors
```

The models load lazily when a command first needs them. The launcher selects
CUDA, then MPS, then CPU; override that choice with `--device cuda`,
`--device mps`, or `--device cpu`. GPU execution uses bf16. CPU execution uses
fp32 and is substantially slower.

Other launch options:

```console
$ uv run python playground.py --arms vec,text --pf 8
$ uv run python playground.py --no-color
$ uv run python playground.py --help
```

Every session is also saved without terminal colors under
`playground_logs/session_<timestamp>.log`. The directory is created on first
launch. Command history is retained in `~/.playground_history`.

## Reference results

These released single-seed results are useful for checking a local setup:

- OOD-600 at pf4: 527/600 byte-exact and 577/600 code-exact, with a mean
  3.73 times fewer decoder context slots.
- Canary development set at pf4: 30/36 byte-exact and 35/36 code-exact.
- OOD code-exact by pooling factor: 577 at pf4, 522 at pf6, 389 at pf8, and
  6 at pf12.
- Fully OOD question answering: 39.0% from latents versus 39.6% from text;
  the two are statistically indistinguishable.

These results cover one seed, Python, and Qwen3-1.7B. They should not be read
as evidence for other languages, seeds, or base models.

## Commands

| Command | Effect |
|---|---|
| `demo` | Run a short guided tour. |
| `load canary <index\|name>` | Load one of the 36 canary functions. Omit the selector for an interactive picker. |
| `load ood <index>` | Load one of the 600 post-cutoff OOD functions. Omit the index for the picker. |
| `load qa <index>` | Load a fully OOD QA record with its gold question and answer. Omit the index for the picker. |
| `paste` | Paste one function, ending with a line containing only `.`. Indented methods are dedented automatically. |
| `show` | Print the current function with line numbers and syntax highlighting. |
| `pf <n>` | Set the latent pooling factor. The trained factors are 4 and 8. |
| `arms <list>` | Select any of `vec`, `text`, and `stock`; for example, `arms vec text`. |
| `recon` | Reconstruct with each selected arm and grade exactness. |
| `compare` | Run the vector arm at pf4, pf6, pf8, and pf12. |
| `sample [count] [temperature]` | Sample several reconstructions and show per-line agreement. Defaults to five samples at temperature 0.8. |
| `ask <question>` | Ask about the loaded function. On a QA record, omit the question to use the gold prompt. |
| `gold` | Show the loaded QA record's gold question and answer. |
| `raw` | Show complete generations. After `recon`, this is a side-by-side diff against the original. |
| `help` | Show the in-program reference. |
| `quit` | Exit. |

A status line above the prompt shows the loaded function, token count, pooling
factor, and active arms. `Ctrl-C` aborts the current generation and returns to
the prompt.

## Paste your own function

```text
pg> paste
paste your function, then a line with just a single .  to finish:
def set_flag(self, name: str) -> None:
    self.flags[name] = True
    self.dirty = True
.
loaded [paste] set_flag
pg> arms vec
pg> recon
```

Pasting at the prompt without first entering `paste` is detected for ordinary
functions, but the explicit command is the reliable choice for long inputs.
A pasted function cannot contain a line consisting only of `.`. Trailing blank
lines are removed.

## Read the reconstruction output

The verdict ladder is:

1. **BYTE-EXACT** — the generated source and original are identical byte for
   byte, with a clean generation boundary.
2. **CODE-EXACT** — bytes differ, but executable code is identical after the
   grader removes comments and docstrings and normalizes trailing whitespace.
3. **MISS** — executable code differs.
4. **UNGRADEABLE** — the original does not parse as a top-level Python
   function, so code-exact grading is unavailable.

`sim%` is a raw character-similarity aid, not an exactness metric. Grading uses
the complete unfiltered generation. Display clipping and collapsed whitespace
do not affect a verdict.

A reconstruction line reports the number of decoder input vectors. The two
learned block-boundary embeddings are included in that count, so the measured
context-slot reduction is slightly below the nominal pooling factor.

Use `raw` to inspect both sources in full:

```text
pg> raw
      ORIGINAL                                  │     vec generation
   1│def round_values(df, *, column: str, dec…  │   1│def round_values(df, *, column: str, dec…
~ 24│    :-----:|:-----:|:-----:                │ 24│    :-----:|-----:|-----:|
```

Changed lines carry a gutter marker, and differing characters are highlighted
when color is enabled.

## Explore pooling factors

`compare` runs one function across four pooling factors:

```text
pg> load canary 5
pg> compare
rate sweep on ConfigurationManager.defaults_
    pf │ vectors │ verdict      │  sim% │  time
     4 │      39 │ BYTE-EXACT   │ 100.0 │  ...
     6 │      27 │ CODE-EXACT   │  99.1 │  ...
     8 │      21 │ CODE-EXACT   │  98.5 │  ...
    12 │      15 │ MISS         │  46.7 │  ...
```

Factors 4 and 8 were used during training. Factor 6 interpolates between them;
pf12 illustrates the sharp fidelity loss at a more aggressive setting.

`sample` complements greedy reconstruction by showing which lines remain
stable across stochastic generations:

```text
pg> sample 5 0.8
  disputed lines (votes across samples):
    line 21: 4× "**Input**" · 1× "** Input**"
5 samples @ temp 0.8: 40/44 lines unanimous, 4 disputed
```

Greedy decoding is used for exactness results. Sampling is exploratory.

## Ask questions

Questions work with any loaded function:

```text
pg> ask what does this function yield?
--- [vec] ---
   This function yields each option, its associated metadata, and the
   corresponding configuration instance.
```

The vector and text arms use the finetuned decoder; the stock arm supplies a
base-model comparison. For released fully OOD evaluation, comprehension from
latents is statistically indistinguishable from comprehension from text.

## Multi-function stacks

Stack mode assembles several functions into one decoder context. Each function
is preceded by a short name header; `recon` selects one member as the target.

| Command | Effect |
|---|---|
| `stack add canary 5` | Add a canary function. `ood`, `qa`, and `paste` are also accepted. |
| `stack show` | Show numbered members and the running token/vector totals. |
| `stack rm <name\|index>` | Remove a member. |
| `stack clear` | Leave stack mode. |
| `recon <name\|index>` | Reconstruct one member from the complete stack context. |
| `ask <question>` | Ask a question over the complete stack. |

Large stacks can be slow. If a generation resembles another member more than
the selected target, the playground calls that out alongside the similarity
scores.

## Notes

- The released checkpoint reads continuous latent vectors only through the
  finetuned decoder. The stock model has no vector arm because it was not
  trained to interpret them.
- Exactness decreases with function length and with pooling factors beyond the
  trained settings.
- Up to three Qwen3-1.7B-sized models may be resident when all arms are active.
  Select fewer arms on memory-constrained hardware.
- Colors disable automatically when output is piped. Set `PG_COLOR=1` to force
  them. Syntax highlighting uses Pygments when available and a standard-library
  fallback otherwise.
- The interactive picker requires a terminal. With piped input, it falls back
  to a numbered list.
- If weights are missing, the playground prints the exact download command
  instead of a traceback.
