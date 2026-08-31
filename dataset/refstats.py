"""Compute reference distributions for corpus validation.

Summarizes length bands and style axes for:
- ``data/csn_python`` training rows, deriving style features from source code;
- ``data/postcut_corpus/functions.jsonl``, deriving ``typed`` consistently.

Method detection is heuristic because CodeSearchNet rows do not retain class
context: a first argument named ``self`` or ``cls`` marks a method.

Output: corpus_meta/ref_stats.json

Run: uv run python dataset/refstats.py
"""
import ast
import json
import warnings
from pathlib import Path

warnings.simplefilter("ignore", SyntaxWarning)
ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "corpus_v2" / "corpus_meta"
BANDS = [(16, 63), (64, 255), (256, 511), (512, 1023), (1024, 4095)]


def band_of(n):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= n <= hi:
            return i
    return 4 if n > 4095 else -1


def style_of(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    n = tree.body[0]
    args = n.args.args
    return {
        "decorated": bool(n.decorator_list),
        "typed": bool(n.returns) or any(a.annotation for a in args),
        "has_docstring": ast.get_docstring(n) is not None,
        "is_async": isinstance(n, ast.AsyncFunctionDef),
        "is_method": bool(args) and args[0].arg in ("self", "cls"),
        "string_share": round(sum(len(x.value) for x in ast.walk(n)
                                  if isinstance(x, ast.Constant) and isinstance(x.value, str)
                                  ) / max(len(code), 1), 4),
    }


def summarize(rows_iter, n_tok_key):
    bands = [0] * 6
    sty = {k: 0.0 for k in ("decorated", "typed", "has_docstring", "is_async",
                            "is_method", "string_share")}
    n = parsed = 0
    for r in rows_iter:
        n += 1
        b = band_of(r[n_tok_key])
        bands[b if b >= 0 else 5] += 1
        s = style_of(r["code"])
        if s:
            parsed += 1
            for k in sty:
                sty[k] += s[k]
    return {
        "n": n, "style_parsed": parsed,
        "bands": {f"{lo}-{hi}": bands[i] for i, (lo, hi) in enumerate(BANDS)},
        "out_of_band": bands[5],
        "style": {k: round(v / max(parsed, 1), 4) for k, v in sty.items()},
    }


out = {}
from datasets import load_from_disk
csn = load_from_disk(str(ROOT / "data" / "csn_python"))["train"]
out["csn_train"] = summarize(
    ({"code": c, "nt": t} for c, t in zip(csn["code"], csn["n_tokens"])), "nt")
print("csn done", flush=True)

out["postcut"] = summarize(
    (json.loads(l) for l in open(ROOT / "data/postcut_corpus/functions.jsonl")), "n_tok")
print("postcut done", flush=True)

json.dump(out, open(META / "ref_stats.json", "w"), indent=1)
print(json.dumps(out, indent=1))
