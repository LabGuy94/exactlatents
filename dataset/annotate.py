"""Corpus sampler stage A: annotate every harvested function row.

Streams all source piles shard by shard in a multiprocessing pool and writes
one compact parquet of per-row metadata per input shard under
``data/corpus_v2/corpus_meta/annot``. The draw stage operates entirely on these
parquets; code text is streamed again only during materialization.

Per row we record:
- identity: pile, shard index, line number (0-based) -> exact reproducibility
- md5(code.strip()) content hash (cross-pile dedup + eval-exclusion recheck)
- skeleton family hash: tokenize-module normalization (identifiers/numbers/
  strings -> placeholders, comments + stmt-leading strings i.e. docstrings
  stripped, keywords/operators/indentation kept)
- junk trips: hard (max line > 1000 chars), soft (mean line > 200, whitespace
  ratio < 5%, unbroken alnum run > 200, string_share > 0.9 AND n_tokens >= 512)
- stub flag: body after docstring is exactly pass / ... / raise NotImplementedError
- style axes + n_tokens + repo carried through from the harvest row
- file_ts clamped to null (-1) when > build time (bogus future mtimes), counted

Run: uv run python dataset/annotate.py [--workers 8] [--limit-shards N]
"""
import argparse
import ast
import hashlib
import io
import json
import keyword
import re
import time
import tokenize
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CV2 = ROOT / "data" / "corpus_v2"
META = CV2 / "corpus_meta"
ANNOT = META / "annot"

BUILD_NOW = 1785993154  # Fixed build epoch for reproducible mtime clamping.

PILES = {0: "stackv3", 1: "starcoder", 2: "starcoder_long", 3: "starcoder_valley"}

_ALNUM_RUN = re.compile(r"[A-Za-z0-9]{201,}")

# regex fallback skeleton (only when tokenize chokes; counted)
_RX_STR = re.compile(r"('''.*?'''|\"\"\".*?\"\"\"|'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\")", re.S)
_RX_CMT = re.compile(r"#[^\n]*")
_RX_NUM = re.compile(r"\b\d[\w.]*\b")
_RX_NAME = re.compile(r"\b[A-Za-z_]\w*\b")


def skeleton_hash(code):
    """(hash, used_fallback). Docstrings = statement-leading string exprs dropped."""
    try:
        out = []
        at_stmt_start = True
        skip_nl = False
        for tk in tokenize.generate_tokens(io.StringIO(code).readline):
            t = tk.type
            if t in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            if t == tokenize.NEWLINE:
                if skip_nl:
                    skip_nl = False
                else:
                    out.append("\n")
                at_stmt_start = True
                continue
            if t == tokenize.STRING and at_stmt_start:
                skip_nl = True          # docstring/bare-string stmt: drop it + its NEWLINE
                at_stmt_start = False
                continue
            if t == tokenize.NAME:
                out.append(tk.string if keyword.iskeyword(tk.string) else "N")
            elif t == tokenize.NUMBER:
                out.append("0")
            elif t == tokenize.STRING:
                out.append("S")
            elif t == tokenize.INDENT:
                out.append(">")
            elif t == tokenize.DEDENT:
                out.append("<")
            else:
                out.append(tk.string)
            at_stmt_start = t in (tokenize.INDENT, tokenize.DEDENT)
        return hashlib.md5("\x00".join(out).encode()).hexdigest(), False
    except Exception:
        s = _RX_STR.sub("S", code)
        s = _RX_CMT.sub("", s)
        s = _RX_NUM.sub("0", s)
        s = _RX_NAME.sub(lambda m: m.group(0) if keyword.iskeyword(m.group(0)) else "N", s)
        s = re.sub(r"[ \t]+", " ", s)
        return hashlib.md5(s.encode()).hexdigest(), True


def is_stub(code):
    """Body after docstring is a single pass / ... / raise NotImplementedError."""
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    body = tree.body[0].body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 1:
        return False
    st = body[0]
    if isinstance(st, ast.Pass):
        return True
    if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and st.value.value is Ellipsis:
        return True
    if isinstance(st, ast.Raise):
        exc = st.exc
        if isinstance(exc, ast.Call):
            exc = exc.func
        return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"
    return False


def annotate_shard(task):
    pile, shard_idx, path = task
    import warnings
    warnings.simplefilter("ignore", SyntaxWarning)  # ast.parse of arbitrary user code
    import pandas as pd
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f):
            r = json.loads(line)
            code = r["code"]
            h = hashlib.md5(code.strip().encode()).digest()
            fam, fam_fb = skeleton_hash(code)

            lines = code.split("\n")
            max_line = max(len(l) for l in lines)
            mean_line = len(code) / max(len(lines), 1)
            n_ws = sum(1 for c in code if c in " \t\n\r")
            ws_ratio = n_ws / max(len(code), 1)
            alnum_run = bool(_ALNUM_RUN.search(code))
            n_tok = r["n_tokens"]
            ss = r.get("string_share", 0.0) or 0.0

            hard_line = max_line > 1000
            soft_mean = mean_line > 200
            soft_ws = ws_ratio < 0.05
            soft_run = alnum_run
            soft_str = ss > 0.9 and n_tok >= 512
            n_soft = soft_mean + soft_ws + soft_run + soft_str
            junk = hard_line or n_soft >= 2

            ts = r.get("file_ts")
            ts_clamped = ts is not None and ts > BUILD_NOW
            rows.append(dict(
                pile=pile, shard=shard_idx, line=line_no,
                h=h, fam=bytes.fromhex(fam), fam_fb=fam_fb,
                repo=r["repo"], n_tokens=n_tok,
                hard_line=hard_line, soft_mean=soft_mean, soft_ws=soft_ws,
                soft_run=soft_run, soft_str=soft_str, junk=junk,
                stub=is_stub(code),
                decorated=bool(r.get("decorated")), typed=bool(r.get("typed")),
                has_docstring=bool(r.get("has_docstring")),
                is_method=bool(r.get("is_method")), is_async=bool(r.get("is_async")),
                string_share=ss,
                file_ts=-1 if (ts is None or ts_clamped) else int(ts),
                ts_clamped=ts_clamped,
            ))
    df = pd.DataFrame(rows)
    out = ANNOT / f"p{pile}_s{shard_idx:04d}.parquet"
    df.to_parquet(out, index=False)
    return (pile, shard_idx, len(df), int(df.fam_fb.sum()))


def shard_tasks():
    """Deterministic (pile, shard_idx, path) list. Shard index = canonical order."""
    tasks = []
    sv3 = []
    for w in sorted((CV2 / "stackv3_raw" / "pod_out").glob("w*"),
                    key=lambda p: int(p.name[1:])):
        sv3 += sorted(w.glob("functions_*.jsonl"))
    for i, p in enumerate(sv3):
        tasks.append((0, i, p))
    for i, p in enumerate(sorted((CV2 / "starcoder" / "functions").glob("shard_*.jsonl"))):
        tasks.append((1, i, p))
    for i, p in enumerate(sorted((CV2 / "starcoder_long" / "functions").glob("shard_*.jsonl"))):
        tasks.append((2, i, p))
    for i, p in enumerate(sorted((CV2 / "starcoder_valley" / "functions").glob("shard_*.jsonl"))):
        tasks.append((3, i, p))
    return tasks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-shards", type=int, default=None)
    args = ap.parse_args()

    ANNOT.mkdir(parents=True, exist_ok=True)
    tasks = shard_tasks()
    if args.limit_shards:
        tasks = tasks[: args.limit_shards]
    # skip already-done shards (resume)
    todo = [t for t in tasks
            if not (ANNOT / f"p{t[0]}_s{t[1]:04d}.parquet").exists()]
    print(f"{len(tasks)} shards total, {len(todo)} to do", flush=True)
    t0 = time.time()
    done_rows = 0
    with Pool(args.workers) as pool:
        for pile, si, n, fb in pool.imap_unordered(annotate_shard, todo):
            done_rows += n
            el = time.time() - t0
            print(f"  p{pile}_s{si:04d}: {n} rows ({fb} tokenize-fallback) "
                  f"| {done_rows} rows in {el/60:.1f}m ({done_rows/max(el,1):.0f}/s)",
                  flush=True)
    print(f"annotate done in {(time.time()-t0)/60:.1f}m")
