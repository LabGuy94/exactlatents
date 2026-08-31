"""Token classification for the trainer's ranking-hinge diagnostics.

Classes: ident_first / ident_rep / keyword / docstring / string / number /
comment / punct / ws / unk. First-occurrence tracking is per function.
"""

import ast
import io
import keyword
import tokenize
from collections import Counter

PROSE = {"docstring", "comment"}


def line_starts(code):
    starts = [0]
    for i, ch in enumerate(code):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def docstring_ranges(code, ls):
    rng = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return rng
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                d = body[0].value
                rng.append((ls[d.lineno - 1] + d.col_offset,
                            ls[d.end_lineno - 1] + d.end_col_offset))
    return rng


def char_classes(code):
    ls = line_starts(code)
    doc = docstring_ranges(code, ls)
    cls = ["ws"] * len(code)
    seen = set()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return ["unk"] * len(code)
    for t in toks:
        if t.start[0] - 1 >= len(ls) or t.end[0] - 1 >= len(ls):
            continue
        s = ls[t.start[0] - 1] + t.start[1]
        e = ls[t.end[0] - 1] + t.end[1]
        if e <= s:
            continue
        if t.type == tokenize.NAME:
            c = "keyword" if keyword.iskeyword(t.string) else \
                ("ident_rep" if t.string in seen else "ident_first")
            seen.add(t.string)
        elif t.type == tokenize.STRING:
            c = "docstring" if any(a <= s and e <= b for a, b in doc) else "string"
        elif t.type == tokenize.NUMBER:
            c = "number"
        elif t.type == tokenize.COMMENT:
            c = "comment"
        elif t.type == tokenize.OP:
            c = "punct"
        else:
            continue
        for i in range(s, min(e, len(code))):
            cls[i] = c
    return cls


def classify_span(cls, s, e):
    counts = Counter(cls[i] for i in range(s, min(e, len(cls))))
    if len(counts) > 1:
        counts.pop("ws", None)
    return counts.most_common(1)[0][0] if counts else "ws"


def token_classes(tok, code):
    """Per-BPE-token class list + offsets for one function."""
    enc = tok(code, add_special_tokens=False, return_offsets_mapping=True)
    cls = char_classes(code)
    return [classify_span(cls, s, e) for s, e in enc["offset_mapping"]], \
        enc["input_ids"], enc["offset_mapping"]
