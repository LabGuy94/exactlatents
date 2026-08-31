"""Versioned byte and code exactness metrics.

Both measures are reported side by side:

``byte_exact``
    The generation reproduces the original byte for byte, with no whitespace
    normalization or prefix-only matching.

``code_exact``
    Docstrings and comments are removed with AST and tokenization support,
    trailing whitespace and blank lines are normalized, and string literals
    and indentation remain significant. Regex removal is deliberately avoided
    because it can consume string data. This prose-blind metric complements
    byte exactness when a generated docstring is paraphrased.

Free-running generation may continue past the function; both metrics handle
that boundary explicitly.
"""

import ast
import io
import tokenize

VERSION = "exactness-v1"


def byte_exact(code: str, gen: str) -> bool:
    """Generation opens with the original, byte for byte, and the boundary is
    clean: the next character (if any) starts a new line — the function ended
    where it should, not mid-token."""
    if not gen.startswith(code):
        return False
    rest = gen[len(code):]
    return rest == "" or rest[0] == "\n" or code.endswith("\n")


def _blank_docstrings(src: str) -> str:
    """Replace docstring constants with pass-equivalent placeholders.
    AST line and column offsets are byte offsets, so this slices bytes."""
    tree = ast.parse(src)
    spans = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (body and isinstance(node, (ast.Module, ast.ClassDef,
                                       ast.FunctionDef, ast.AsyncFunctionDef))
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            d = body[0]
            spans.append((d.lineno, d.col_offset, d.end_lineno, d.end_col_offset))
    if not spans:
        return src
    lines = [l.encode() for l in src.split("\n")]
    for l1, c1, l2, c2 in spans:
        if l1 == l2:
            lines[l1 - 1] = lines[l1 - 1][:c1] + b"pass" + lines[l1 - 1][c2:]
        else:
            lines[l1 - 1] = lines[l1 - 1][:c1] + b"pass"
            for i in range(l1, l2 - 1):
                lines[i] = b""
            lines[l2 - 1] = lines[l2 - 1][c2:]
    return "\n".join(l.decode() for l in lines)


def normalize_code(src: str):
    """Docstrings blanked, comments dropped (tokenize), trailing whitespace
    stripped, blank lines collapsed. Returns None if src does not parse —
    unparseable output is never code-exact."""
    try:
        src = _blank_docstrings(src)
        out_lines = {}
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                l = tok.start[0]
                line = src.split("\n")[l - 1]
                out_lines[l] = line[: tok.start[1]]
        lines = src.split("\n")
        for l, repl in out_lines.items():
            lines[l - 1] = repl
        cleaned = [l.rstrip() for l in lines]
        return "\n".join(l for l in cleaned if l != "")
    except (SyntaxError, tokenize.TokenError, IndentationError, IndexError):
        return None


def code_exact(code: str, gen: str) -> bool:
    """Prose-blind equality. The gen stream has no marked end, so try cut
    points at newlines within +-25% of the original length and accept the
    first parse whose normalized form matches."""
    ncode = normalize_code(code)
    if ncode is None:
        return False
    lo, hi = int(len(code) * 0.75), min(len(gen), len(code) + max(200, len(code) // 4))
    cuts = [i for i, ch in enumerate(gen[:hi]) if ch == "\n" and i >= lo]
    cuts.append(hi)
    for cut in cuts:
        ngen = normalize_code(gen[:cut])
        if ngen is not None and ngen == ncode:
            return True
    return False
