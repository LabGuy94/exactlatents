"""AST span segmentation for per-span allocation.

Segments a Python function into typed character ranges. The AST proposes
boundaries, the surprisal column estimates which content is expensive, and
training decides the allocation. The optional annealed allocation objective
uses these spans during training; inference does not.

Span kinds:
  sig       — def line(s): name + parameters + return annotation
  docstring — the docstring constant, if any
  literal   — string/number constants outside the docstring (merged per line)
  body      — everything else

Unparseable code degrades gracefully to one ``body`` span covering the full
input.
"""

import ast
import re

# Exemplar-like fragments inside prose include timestamps, version strings,
# hexadecimal identifiers, URLs, and doctest numerics. Although their context
# is prose, their exact values behave like literals, so allocation can classify
# them separately.
EXEMPLAR_PAT = re.compile(
    r"https?://\S+"                                   # URLs
    r"|\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"  # dates/timestamps
    r"|\b\d+\.\d+\.\d+(?:\.\d+)*\b"                   # dotted versions/builds
    r"|\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{8,}\b"        # hex / long ids
    r"|>>> .+"                                        # doctest input lines
    r"|\barray\(\[[^)]*\)"                            # doctest numeric arrays
    r"|\b\d{5,}\b"                                    # long bare numbers
)


def _split_exemplars(code, s, e, kind):
    """Split a prose span on exemplar matches -> [(kind|'exemplar', s, e), ...]."""
    out, cur = [], s
    for m in EXEMPLAR_PAT.finditer(code[s:e]):
        ms, me = s + m.start(), s + m.end()
        if ms > cur:
            out.append((kind, cur, ms))
        out.append(("exemplar", ms, me))
        cur = me
    if cur < e:
        out.append((kind, cur, e))
    return out or [(kind, s, e)]


def _seg(lines, a, b):
    """(lineno, col) pair -> absolute char offsets into '\n'.join(lines)."""
    starts = []
    off = 0
    for ln in lines:
        starts.append(off)
        off += len(ln) + 1
    la, ca = a
    lb, cb = b
    return starts[la - 1] + ca, starts[lb - 1] + cb


def segment(code):
    """-> list of (kind, char_start, char_end), non-overlapping, sorted,
    covering [0, len(code)). Falls back to [('body', 0, len(code))]."""
    whole = [("body", 0, len(code))]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return whole
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None or not fn.body:
        return whole
    lines = code.split("\n")

    marks = []  # (start, end, kind) claims, later flattened with body filling gaps

    # signature: from 'def' line to the end of the last argument / return annot
    sig_end_node = fn.returns or (fn.args.args[-1] if fn.args.args else None)
    first_stmt = fn.body[0]
    sig_start, _ = _seg(lines, (fn.lineno, fn.col_offset), (fn.lineno, fn.col_offset))
    body_start, _ = _seg(lines, (first_stmt.lineno, first_stmt.col_offset),
                         (first_stmt.lineno, first_stmt.col_offset))
    # simplest robust signature = everything before the first body statement
    marks.append((sig_start, body_start, "sig"))

    doc_node = None
    if (isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)):
        doc_node = first_stmt
        s, e = _seg(lines, (first_stmt.lineno, first_stmt.col_offset),
                    (first_stmt.end_lineno, first_stmt.end_col_offset))
        marks.append((s, e, "docstring"))

    for node in ast.walk(fn):
        if node is doc_node or (doc_node and node is doc_node.value):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, complex)):
            if node.end_lineno is None:
                continue
            s, e = _seg(lines, (node.lineno, node.col_offset),
                        (node.end_lineno, node.end_col_offset))
            if e > s:
                marks.append((s, e, "literal"))

    # flatten: sort claims, drop overlaps (first claim wins: sig > docstring > literal
    # by construction order for ties), fill gaps with body
    marks.sort(key=lambda m: (m[0], m[1]))
    spans, cur = [], 0
    for s, e, kind in marks:
        if s < cur:  # overlapped by an earlier claim
            if e <= cur:
                continue
            s = cur
        if s > cur:
            spans.append(("body", cur, s))
        if kind == "docstring":
            spans.extend(_split_exemplars(code, s, e, "docstring"))
        else:
            spans.append((kind, s, e))
        cur = e
    if cur < len(code):
        spans.append(("body", cur, len(code)))
    return spans


def span_masses(code, spans, surp_row=None, tok=None):
    """Per-span weight for slot allocation: surprisal mass if a per-token
    surprisal row + tokenizer (with offset mapping) are given, else char mass.
    Returns list of floats aligned with spans, summing to 1.0 (uniform on
    degenerate input)."""
    if surp_row is not None and tok is not None:
        enc = tok(code, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        n = min(len(offs), len(surp_row))
        masses = []
        for kind, s, e in spans:
            m = sum(float(surp_row[i]) for i in range(n)
                    if offs[i][0] < e and offs[i][1] > s)
            masses.append(m)
    else:
        masses = [float(e - s) for _, s, e in spans]
    total = sum(masses)
    if total <= 0:
        return [1.0 / len(spans)] * len(spans)
    return [m / total for m in masses]
