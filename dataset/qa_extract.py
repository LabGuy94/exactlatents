"""Extract deterministic QA intents from corpus functions with an AST engine.

Produces ``(function_id, intent, question_seed, answer)`` triples without model
calls. ``question_seed`` is a structured dictionary of the intent identifier
and fact parameters; natural-language phrasing is a separate stage.

The registry maps each intent identifier to applicability and extraction
functions. Groups A, B, C, E/F, G, H, and I are implemented here. Group D and
runnable C2 intents are implemented in ``dataset/qa_exec.py``.

Answer-leak rule: for non-MCQ intents the question_seed must never contain the
answer string. Enforced at emission via leak_check(); MCQ seeds carry options
by design and are exempt. See COVERAGE_REPORT for intents where leakage is
structurally inherent (none found; D2 carries the *output* in the seed and the
*input* as answer, which is the intent's definition, not a leak).

Line numbers everywhere are 1-based and relative to the code snippet (the
corpus `code` field), never the original file. Span answers (group B) are
byte-exact substrings of the ORIGINAL code field, verified by `in` check at
emission; facts failing verification are dropped and counted.
"""

from __future__ import annotations

import ast
import builtins as _builtins_mod
import hashlib
import random
import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Optional

# --------------------------------------------------------------------------
# Fact / registry plumbing
# --------------------------------------------------------------------------


@dataclass
class Fact:
    intent: str
    question_seed: dict
    answer: Optional[str]  # None only for needs_luna slots
    answer_type: str  # substring|number|yesno|name|letter|line_ref|list|text|null
    needs_luna: bool = False
    needs_phrasing: bool = False
    held_out: bool = False
    mcq: bool = False  # exempt from leak check
    leak_by_design: bool = False  # seed carries a paired value by definition
    spec_question: bool = False  # implementation had to interpret the catalog


@dataclass
class IntentSpec:
    iid: str
    group: str
    kind: str  # "fn" (per-function) or "stack" (multi-function)
    applicable: Callable  # (ctx) -> bool          [fn kind]
    extract: Callable  # (ctx) -> list[Fact]       [fn kind]
    held_out: bool = False
    needs_luna: bool = False
    description: str = ""


REGISTRY: dict[str, IntentSpec] = {}


def register(iid, group, *, kind="fn", held_out=False, needs_luna=False, description=""):
    def deco(pair):
        applicable, extract = pair
        REGISTRY[iid] = IntentSpec(
            iid, group, kind, applicable, extract, held_out, needs_luna, description
        )
        return pair

    return deco


# --------------------------------------------------------------------------
# Function context
# --------------------------------------------------------------------------

BUILTIN_NAMES = frozenset(dir(_builtins_mod))

# tokens too trivial for the leak check to be meaningful
_TRIVIAL_ANSWERS = frozenset(
    {"yes", "no", "none", "then", "else", "0", "1", "2", "3", "true", "false"}
)


class ParseFailure(Exception):
    pass


@dataclass
class FnCtx:
    row: dict
    row_idx: int  # index in the FULL train split
    code: str  # original corpus bytes (answers verified against this)
    norm_code: str  # parseable text (== code unless indent-normalized)
    normalized: bool
    lines: list[str]
    tree: ast.Module
    fn: ast.AST  # FunctionDef | AsyncFunctionDef
    rng: random.Random
    self_contained: Optional[bool] = None  # set by driver from qa_exec.screen
    pools: Optional[dict] = None  # corpus-level distractor pools
    _cache: dict = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------
    def seg(self, node) -> Optional[str]:
        try:
            return ast.get_source_segment(self.norm_code, node)
        except Exception:
            return None

    def span_ok(self, s: Optional[str]) -> bool:
        """Byte-exact-substring verification against the ORIGINAL code."""
        return bool(s) and s in self.code

    def body_walk(self):
        if "walk" not in self._cache:
            self._cache["walk"] = list(ast.walk(self.fn))
        return self._cache["walk"]

    def params(self):
        """All parameter names in signature order."""
        if "params" not in self._cache:
            a = self.fn.args
            names = [p.arg for p in a.posonlyargs + a.args]
            if a.vararg:
                names.append(a.vararg.arg)
            names += [p.arg for p in a.kwonlyargs]
            if a.kwarg:
                names.append(a.kwarg.arg)
            self._cache["params"] = names
        return self._cache["params"]

    def returns_with_value(self):
        if "rets" not in self._cache:
            self._cache["rets"] = [
                n for n in self.body_walk()
                if isinstance(n, ast.Return) and n.value is not None
                and not _in_nested_fn(self.fn, n)
            ]
        return self._cache["rets"]

    def docstring_node(self):
        b = self.fn.body
        if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                and isinstance(b[0].value.value, str):
            return b[0].value
        return None


def _in_nested_fn(outer, node):
    """True if node sits inside a def/lambda nested within `outer`.
    Computed by a parent map (built lazily per outer fn)."""
    pm = getattr(outer, "_qa_parent_map", None)
    if pm is None:
        pm = {}
        for parent in ast.walk(outer):
            for child in ast.iter_child_nodes(parent):
                pm[id(child)] = parent
        outer._qa_parent_map = pm
    cur = pm.get(id(node))
    while cur is not None and cur is not outer:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return True
        cur = pm.get(id(cur))
    return False


def _normalize_for_parse(code: str) -> tuple[str, bool]:
    """Corpus quirk: ~5% of rows are decorated methods where decorators sit at
    col 0 but the def keeps class indentation ('unexpected indent'). Fix by
    stripping the def line's indent from the def line onward; decorator lines
    untouched. Span answers are verified against the ORIGINAL code afterwards,
    so normalization can only drop facts, never corrupt them."""
    try:
        ast.parse(code)
        return code, False
    except SyntaxError:
        pass
    lines = code.split("\n")
    def_i = None
    for i, ln in enumerate(lines):
        if re.match(r"\s+(async\s+)?def\s", ln):
            def_i = i
            break
    if def_i is None:
        raise ParseFailure("unparseable, no indented def")
    indent = len(lines[def_i]) - len(lines[def_i].lstrip())
    fixed = lines[:def_i] + [
        ln[indent:] if ln[:indent].strip() == "" else ln for ln in lines[def_i:]
    ]
    norm = "\n".join(fixed)
    try:
        ast.parse(norm)
    except SyntaxError as e:
        raise ParseFailure(f"unparseable after normalize: {e}") from None
    return norm, True


def make_ctx(row: dict, row_idx: int, pools=None, self_contained=None) -> FnCtx:
    code = row["code"]
    norm, normalized = _normalize_for_parse(code)
    tree = ast.parse(norm)
    fn = next(
        (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if fn is None:
        raise ParseFailure("no function def at top level")
    seed = int(hashlib.md5(row["qualified"].encode()).hexdigest()[:12], 16)
    return FnCtx(
        row=row, row_idx=row_idx, code=code, norm_code=norm, normalized=normalized,
        lines=code.split("\n"), tree=tree, fn=fn, rng=random.Random(seed),
        self_contained=self_contained, pools=pools,
    )


# --------------------------------------------------------------------------
# Leak check
# --------------------------------------------------------------------------


def leak_check(fact: Fact) -> bool:
    """True = fact is clean. Non-MCQ seeds must not contain the answer."""
    if fact.mcq or fact.leak_by_design or fact.answer is None:
        return True
    a = fact.answer.strip()
    if a.lower() in _TRIVIAL_ANSWERS or len(a) < 3:
        return True  # yes/no/small-number answers: seed containment meaningless

    def walk(v):
        if isinstance(v, str):
            return a in v
        if isinstance(v, (int, float)):
            return a == str(v)
        if isinstance(v, dict):
            return any(walk(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return any(walk(x) for x in v)
        return False

    return not any(walk(v) for k, v in fact.question_seed.items() if k != "intent")


# --------------------------------------------------------------------------
# Shared analyses
# --------------------------------------------------------------------------

MUTATING_METHODS = frozenset(
    "append extend insert remove pop clear sort reverse update add discard "
    "setdefault popitem appendleft extendleft".split()
)


def mutated_params(ctx: FnCtx) -> list[str]:
    """Params with AST evidence of in-place mutation (attr/subscript assign,
    mutating method call). Plain `p += x` excluded (rebinding for immutables)."""
    pset = set(ctx.params())
    hit = set()
    for n in ctx.body_walk():
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                base = _attr_sub_base(t)
                if base in pset:
                    hit.add(base)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in MUTATING_METHODS and isinstance(n.func.value, ast.Name) \
                    and n.func.value.id in pset:
                hit.add(n.func.value.id)
    return [p for p in ctx.params() if p in hit]


def _attr_sub_base(t):
    """Name at the base of an Attribute/Subscript chain, else None."""
    while isinstance(t, (ast.Attribute, ast.Subscript)):
        t = t.value
    return t.id if isinstance(t, ast.Name) else None


def assigned_names(ctx: FnCtx) -> dict[str, list[int]]:
    """simple-Name assignment targets -> [linenos] (excl. nested fns)."""
    out: dict[str, list[int]] = {}
    for n in ctx.body_walk():
        if _in_nested_fn(ctx.fn, n):
            continue
        tgts = []
        if isinstance(n, ast.Assign):
            tgts = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            tgts = [n.target] if (not isinstance(n, ast.AnnAssign) or n.value) else []
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            tgts = [n.target]
        elif isinstance(n, ast.withitem) and n.optional_vars:
            tgts = [n.optional_vars]
        for t in tgts:
            for leaf in ast.walk(t):
                if isinstance(leaf, ast.Name):
                    out.setdefault(leaf.id, []).append(getattr(n, "lineno", leaf.lineno))
    return out


def loaded_names(ctx: FnCtx) -> set[str]:
    return {
        n.id for n in ctx.body_walk()
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def explicit_raises(ctx: FnCtx) -> list[str]:
    """Exception class names from explicit `raise` statements, source order."""
    out = []
    for n in ctx.body_walk():
        if isinstance(n, ast.Raise) and n.exc is not None:
            e = n.exc
            if isinstance(e, ast.Call):
                e = e.func
            name = None
            if isinstance(e, ast.Name):
                name = e.id
            elif isinstance(e, ast.Attribute):
                name = e.attr
            if name and name not in out:
                out.append(name)
    return out


def _first_docstring_line(ctx: FnCtx) -> Optional[str]:
    node = ctx.docstring_node()
    if node is None:
        return None
    for ln in node.value.split("\n"):
        if ln.strip():
            return ln.strip()
    return None


# ==========================================================================
# GROUP A — identity & structure (12 intents)
# ==========================================================================


def _a1(ctx):
    name = ctx.fn.name
    return [Fact("A1", {"intent": "A1"}, name, "name")]


register("A1", "A", description="name recall")((lambda ctx: True, _a1))


def _a2(ctx):
    return [Fact("A2", {"intent": "A2"}, str(len(ctx.params())), "number")]


register("A2", "A", description="param count")((lambda ctx: True, _a2))


def _a3(ctx):
    return [Fact("A3", {"intent": "A3"}, ", ".join(ctx.params()), "list")]


register("A3", "A", description="param names in order")(
    (lambda ctx: len(ctx.params()) > 0, _a3)
)


def _defaults_map(ctx):
    a = ctx.fn.args
    pos = a.posonlyargs + a.args
    out = {}
    for p, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
        out[p.arg] = d
    for p, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None:
            out[p.arg] = d
    return out


def _a4(ctx):
    dm = _defaults_map(ctx)
    facts = []
    for p in ctx.rng.sample(sorted(dm), min(2, len(dm))):
        src = ctx.seg(dm[p])
        if src and len(src) <= 80:
            facts.append(Fact("A4", {"intent": "A4", "param": p}, src, "substring"))
    return facts


register("A4", "A", description="default value of param P")(
    (lambda ctx: bool(_defaults_map(ctx)), _a4)
)


def _a5(ctx):
    a = ctx.fn.args
    feats = {
        "*args": a.vararg is not None,
        "**kwargs": a.kwarg is not None,
        "keyword-only params": bool(a.kwonlyargs),
    }
    feat = ctx.rng.choice(sorted(feats))
    return [Fact("A5", {"intent": "A5", "feature": feat},
                 "yes" if feats[feat] else "no", "yesno")]


register("A5", "A", description="*args/**kwargs/kw-only presence")(
    (lambda ctx: True, _a5)
)


def _a6(ctx):
    src = ctx.seg(ctx.fn.returns)
    if not src or len(src) > 80:
        return []
    return [Fact("A6", {"intent": "A6"}, src, "substring")]


register("A6", "A", description="return annotation")(
    (lambda ctx: ctx.fn.returns is not None, _a6)
)


def _annotated_params(ctx):
    a = ctx.fn.args
    return {p.arg: p.annotation for p in a.posonlyargs + a.args + a.kwonlyargs
            if p.annotation is not None}


def _a7(ctx):
    am = _annotated_params(ctx)
    facts = []
    for p in ctx.rng.sample(sorted(am), min(2, len(am))):
        src = ctx.seg(am[p])
        if src and len(src) <= 80:
            facts.append(Fact("A7", {"intent": "A7", "param": p}, src, "substring"))
    return facts


register("A7", "A", description="param annotation of P")(
    (lambda ctx: bool(_annotated_params(ctx)), _a7)
)


def _a8(ctx):
    decs = [ctx.seg(d) for d in ctx.fn.decorator_list]
    decs = [d for d in decs if d and len(d) <= 60]
    if not decs or len(decs) != len(ctx.fn.decorator_list):
        return []
    return [Fact("A8", {"intent": "A8"}, ", ".join(decs), "list")]


register("A8", "A", description="decorator inventory")(
    (lambda ctx: bool(ctx.fn.decorator_list), _a8)
)


def _a9(ctx):
    ans = "yes" if isinstance(ctx.fn, ast.AsyncFunctionDef) else "no"
    return [Fact("A9", {"intent": "A9"}, ans, "yesno")]


register("A9", "A", description="is-async")((lambda ctx: True, _a9))


def _a10(ctx):
    facts = [Fact("A10", {"intent": "A10", "aspect": "is_method"},
                  "yes" if ctx.row["is_method"] else "no", "yesno")]
    if ctx.row["is_method"] and "." in ctx.row["func_name"]:
        cls = ctx.row["func_name"].rsplit(".", 2)[-2]
        facts.append(Fact("A10", {"intent": "A10", "aspect": "class_name"},
                          cls, "name"))
    return facts


register("A10", "A", description="is-method / class membership")(
    (lambda ctx: True, _a10)
)


def _count_kinds(ctx):
    walk = [n for n in ctx.body_walk() if not _in_nested_fn(ctx.fn, n)]
    kinds = {
        "return statements": sum(isinstance(n, ast.Return) for n in walk),
        "if statements": sum(isinstance(n, ast.If) for n in walk),
        "loops": sum(isinstance(n, (ast.For, ast.While, ast.AsyncFor)) for n in walk),
        "string literals": len(_string_literals(ctx)),
    }
    called = sorted(
        {n.func.id for n in walk
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    )
    return kinds, called, walk


def _a11(ctx):
    kinds, called, walk = _count_kinds(ctx)
    facts = []
    for kind in ctx.rng.sample(sorted(kinds), 2):
        facts.append(Fact("A11", {"intent": "A11", "count_of": kind},
                          str(kinds[kind]), "number"))
    if called:
        y = ctx.rng.choice(called)
        c = sum(1 for n in walk if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == y)
        facts.append(Fact("A11", {"intent": "A11", "count_of": "calls", "callee": y},
                          str(c), "number"))
    return facts


register("A11", "A", description="count of X")((lambda ctx: True, _a11))


def _string_literals(ctx):
    """String constants in source order, excluding the docstring and f-string
    interior parts (answers must be literals as written)."""
    doc = ctx.docstring_node()
    inside_joined = set()
    for n in ctx.body_walk():
        if isinstance(n, ast.JoinedStr):
            for c in ast.walk(n):
                inside_joined.add(id(c))
    out = []
    for n in ctx.body_walk():
        if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                and n is not doc and id(n) not in inside_joined:
            out.append(n)
    out.sort(key=lambda n: (n.lineno, n.col_offset))
    return out


def _a12(ctx):
    lits = _string_literals(ctx)
    usable = [(i, n) for i, n in enumerate(lits) if 0 < len(n.value) <= 120]
    if not usable:
        return []
    i, n = ctx.rng.choice(usable)
    if not ctx.span_ok(n.value):
        return []  # escaped/continued literal: value differs from source bytes
    return [Fact("A12", {"intent": "A12", "n": i + 1}, n.value, "substring")]


register("A12", "A", description="nth string literal")(
    (lambda ctx: bool(_string_literals(ctx)), _a12)
)


# ==========================================================================
# GROUP B — span grounding (6 intents; byte-exact, verified by slicing)
# ==========================================================================


def _fn_line_range(ctx):
    return ctx.fn.lineno, ctx.fn.end_lineno


def _b1(ctx):
    lo, hi = _fn_line_range(ctx)
    cand = [i for i in range(lo, min(hi, len(ctx.lines)) + 1)
            if len(ctx.lines[i - 1].strip()) > 3]
    if not cand:
        return []
    n = ctx.rng.choice(cand)
    line = ctx.lines[n - 1]
    if not ctx.span_ok(line):
        return []
    return [Fact("B1", {"intent": "B1", "line": n}, line, "substring")]


register("B1", "B", description="quote line N exactly")((lambda ctx: True, _b1))


def _conds(ctx):
    ifs, whiles = [], []
    for n in ctx.body_walk():
        if _in_nested_fn(ctx.fn, n):
            continue
        if isinstance(n, ast.If):
            ifs.append(n)
        elif isinstance(n, ast.While):
            whiles.append(n)
    ifs.sort(key=lambda n: (n.lineno, n.col_offset))
    whiles.sort(key=lambda n: (n.lineno, n.col_offset))
    return ifs, whiles


def _b2(ctx):
    ifs, whiles = _conds(ctx)
    opts = [("if", ifs)] if ifs else []
    if whiles:
        opts.append(("while", whiles))
    if not opts:
        return []
    construct, nodes = ctx.rng.choice(opts)
    k = ctx.rng.randrange(len(nodes))
    src = ctx.seg(nodes[k].test)
    if not ctx.span_ok(src) or len(src) > 160:
        return []
    return [Fact("B2", {"intent": "B2", "construct": construct, "k": k + 1},
                 src, "substring")]


register("B2", "B", description="quote condition of kth if/while")(
    (lambda ctx: any(_conds(ctx)), _b2)
)


def _b3(ctx):
    rets = ctx.returns_with_value()
    if not rets:
        return []
    k = ctx.rng.randrange(len(rets))
    src = ctx.seg(rets[k].value)
    if not ctx.span_ok(src) or len(src) > 160:
        return []
    return [Fact("B3", {"intent": "B3", "k": k + 1}, src, "substring")]


register("B3", "B", description="quote kth return expression")(
    (lambda ctx: bool(ctx.returns_with_value()), _b3)
)


def _b4(ctx):
    first = _first_docstring_line(ctx)
    if not first or not ctx.span_ok(first) or len(first) > 200:
        return []
    return [Fact("B4", {"intent": "B4"}, first, "substring")]


register("B4", "B", description="quote docstring first line")(
    (lambda ctx: ctx.docstring_node() is not None, _b4)
)


def _span_constructs(ctx):
    out = []
    for n in ctx.body_walk():
        if _in_nested_fn(ctx.fn, n):
            continue
        if isinstance(n, (ast.For, ast.AsyncFor)):
            out.append(("for-loop body", n.body))
        elif isinstance(n, ast.While):
            out.append(("while-loop body", n.body))
        elif isinstance(n, ast.Try):
            out.append(("try block", n.body))
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            out.append(("with block", n.body))
    return out


def _b5(ctx):
    cons = _span_constructs(ctx)
    if not cons:
        return []
    # kth occurrence of the chosen label
    label, body = ctx.rng.choice(cons)
    k = 1 + sum(1 for l, b in cons if l == label and b[0].lineno < body[0].lineno)
    lo = body[0].lineno
    hi = max(s.end_lineno for s in body)
    return [Fact("B5", {"intent": "B5", "construct": label, "k": k},
                 f"L{lo}-L{hi}", "line_ref")]


register("B5", "B", description="line span of construct X")(
    (lambda ctx: bool(_span_constructs(ctx)), _b5)
)

_B6_CONSTRUCTS = {
    "with statement": (ast.With, ast.AsyncWith),
    "while loop": (ast.While,),
    "for loop": (ast.For, ast.AsyncFor),
    "try/except": (ast.Try,),
    "lambda": (ast.Lambda,),
    "assert statement": (ast.Assert,),
    "raise statement": (ast.Raise,),
    "yield": (ast.Yield, ast.YieldFrom),
    "list comprehension": (ast.ListComp,),
}


def _absent_constructs(ctx):
    present = tuple(type(n) for n in ctx.body_walk())
    return [name for name, types in _B6_CONSTRUCTS.items()
            if not any(isinstance(n, types) for n in ctx.body_walk())]


def _b6(ctx):
    absent = _absent_constructs(ctx)
    if not absent:
        return []
    name = ctx.rng.choice(absent)
    return [Fact("B6", {"intent": "B6", "construct": name}, "no", "yesno")]


register("B6", "B", description="absence negative: construct not present")(
    (lambda ctx: bool(_absent_constructs(ctx)), _b6)
)


# ==========================================================================
# GROUP C — behavior & semantics (8 intents)
# ==========================================================================


def _c1(ctx):
    return [Fact("C1", {"intent": "C1"}, None, "null", needs_luna=True)]


register("C1", "C", needs_luna=True,
         description="what-does-it-return-conceptually (LUNA)")(
    (lambda ctx: len(ctx.fn.body) > 1, _c1)
)


def _c2(ctx):
    # runnable functions get an EXEC C2 from qa_exec; here only the LUNA slot
    if ctx.self_contained:
        return []
    seqish = _empty_situation_params(ctx)
    if not seqish:
        return []
    p = ctx.rng.choice(seqish)
    return [Fact("C2", {"intent": "C2", "param": p, "situation": "empty"},
                 None, "null", needs_luna=True)]


def _empty_situation_params(ctx):
    """Params that plausibly hold a sequence/mapping/str (annotation or usage)."""
    out = []
    am = _annotated_params(ctx)
    iterated = set()
    for n in ctx.body_walk():
        if isinstance(n, (ast.For, ast.AsyncFor)) and isinstance(n.iter, ast.Name):
            iterated.add(n.iter.id)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id == "len" and n.args \
                and isinstance(n.args[0], ast.Name):
            iterated.add(n.args[0].id)
    for p in ctx.params():
        if p in ("self", "cls"):
            continue
        ann = am.get(p)
        ann_src = (ast.unparse(ann).lower() if ann is not None else "")
        if p in iterated or any(
            t in ann_src for t in ("list", "dict", "str", "sequence", "iterable",
                                   "set", "tuple", "mapping")
        ):
            out.append(p)
    return out


register("C2", "C", description="situation-scoped behavior (EXEC else LUNA)")(
    (lambda ctx: bool(_empty_situation_params(ctx)), _c2)
)


def _c3(ctx):
    # Purpose distractors would require generated purpose texts -> LUNA in this
    # build; MCQ variant deferred to the phrasing stage. spec_question noted.
    return [Fact("C3", {"intent": "C3"}, None, "null", needs_luna=True,
                 spec_question=True)]


register("C3", "C", needs_luna=True, description="purpose/why (LUNA or MCQ)")(
    (lambda ctx: len(ctx.fn.body) > 1, _c3)
)


def _c4_properties(ctx):
    props = {}
    props["mutates an argument"] = "yes" if mutated_params(ctx) else "no"
    props["recursive"] = (
        "yes" if any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == ctx.fn.name for n in ctx.body_walk()
        ) else "no"
    )
    # can-return-None (AST proxy)
    rets = [n for n in ctx.body_walk() if isinstance(n, ast.Return)
            and not _in_nested_fn(ctx.fn, n)]
    bare = any(r.value is None or (isinstance(r.value, ast.Constant)
                                   and r.value.value is None) for r in rets)
    fallthrough = not isinstance(ctx.fn.body[-1], (ast.Return, ast.Raise))
    if not rets:
        props["can return None"] = "yes"
    elif bare or fallthrough:
        props["can return None"] = "yes"
    elif all(not (isinstance(r.value, ast.Constant) and r.value.value is None)
             for r in rets) and not fallthrough:
        props["can return None"] = "no"
    return props


def _c4(ctx):
    props = _c4_properties(ctx)
    name = ctx.rng.choice(sorted(props))
    return [Fact("C4", {"intent": "C4", "property": name}, props[name], "yesno")]


register("C4", "C", description="property verification yes/no (AST proxies)")(
    (lambda ctx: True, _c4)
)


def _mcq(ctx, iid, correct, distractors, extra_seed=None, spec_question=False):
    opts = [correct] + distractors[:3]
    if len(opts) < 4 or len(set(opts)) < 4:
        return []
    ctx.rng.shuffle(opts)
    letter = "ABCD"[opts.index(correct)]
    seed = {"intent": iid, "options": opts}
    if extra_seed:
        seed.update(extra_seed)
    return [Fact(iid, seed, letter, "letter", mcq=True, spec_question=spec_question)]


def _c5(ctx):
    # Best faithful reading without an LLM: "which description matches this
    # code" MCQ; correct = own docstring first line, distractors = docstring
    # first lines of other corpus functions (deterministic pool draw).
    first = _first_docstring_line(ctx)
    if not first or len(first) < 15 or not ctx.pools:
        return []
    pool = ctx.pools.get("doc_first_lines", [])
    if len(pool) < 10:
        return []
    distractors = []
    for j in range(6):
        cand = pool[ctx.rng.randrange(len(pool))]
        if cand != first and cand not in distractors:
            distractors.append(cand)
        if len(distractors) == 3:
            break
    return _mcq(ctx, "C5", first, distractors, spec_question=True)


register("C5", "C", description="algorithm identification (MCQ)")(
    (lambda ctx: _first_docstring_line(ctx) is not None, _c5)
)


def _c6(ctx):
    raises = explicit_raises(ctx)
    if not raises:
        return []
    return [Fact("C6", {"intent": "C6"}, ", ".join(raises), "list")]


register("C6", "C", description="raises-inventory from explicit raise")(
    (lambda ctx: bool(explicit_raises(ctx)), _c6)
)

_C7_ROLES = [
    "accumulator", "loop counter", "boolean flag", "container being built",
    "result holder", "fixed value (constant)", "temporary",
]


def _c7_classify(ctx):
    """Deterministic variable-role classification (Sajaniemi-style).
    Returns {var: role} for vars with a single unambiguous role."""
    assigned = assigned_names(ctx)
    pset = set(ctx.params())
    loops = [n for n in ctx.body_walk()
             if isinstance(n, (ast.For, ast.While, ast.AsyncFor))]
    in_loop = set()
    for lp in loops:
        for n in ast.walk(lp):
            in_loop.add(id(n))
    ret_names = set()
    for r in ctx.returns_with_value():
        if isinstance(r.value, ast.Name):
            ret_names.add(r.value.id)
    roles = {}
    for var, lines in assigned.items():
        if var in pset or var.startswith("_"):
            continue
        evidence = set()
        for n in ctx.body_walk():
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) \
                    and n.target.id == var and id(n) in in_loop:
                op_arith = isinstance(n.op, (ast.Add, ast.Sub, ast.Mult))
                evidence.add("accumulator" if op_arith else "container being built")
            if isinstance(n, (ast.For, ast.AsyncFor)) and isinstance(n.target, ast.Name) \
                    and n.target.id == var:
                evidence.add("loop counter")
            if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == var for t in n.targets
            ):
                if isinstance(n.value, ast.Constant) and isinstance(n.value.value, bool):
                    evidence.add("boolean flag")
                elif isinstance(n.value, (ast.List, ast.Dict, ast.Set)) \
                        and not n.value.__dict__.get("elts", True):
                    pass
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr in ("append", "extend", "add", "update") \
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == var \
                    and id(n) in in_loop:
                evidence.add("container being built")
        if not evidence and var in ret_names and len(lines) >= 1:
            evidence.add("result holder")
        if not evidence and len(lines) == 1 and var not in ret_names:
            # assigned once from a constant, read later -> fixed value
            for n in ctx.body_walk():
                if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == var for t in n.targets
                ) and isinstance(n.value, ast.Constant):
                    evidence.add("fixed value (constant)")
        if len(evidence) == 1:
            roles[var] = evidence.pop()
    return roles


def _c7(ctx):
    roles = _c7_classify(ctx)
    if not roles:
        return []
    var = ctx.rng.choice(sorted(roles))
    correct = roles[var]
    distractors = [r for r in _C7_ROLES if r != correct]
    ctx.rng.shuffle(distractors)
    return _mcq(ctx, "C7", correct, distractors[:3], extra_seed={"var": var})


register("C7", "C", description="role of variable V (MCQ)")(
    (lambda ctx: bool(_c7_classify(ctx)), _c7)
)


# ==========================================================================
# GROUP E/F — control & data flow (7 intents; F2 HELD OUT)
# ==========================================================================


def _guard_conditions(ctx):
    """Early `if <cond>: return/return None/raise` guards in the first few
    top-level statements."""
    out = []
    body = ctx.fn.body
    start = 1 if ctx.docstring_node() is not None else 0
    for stmt in body[start:start + 4]:
        if isinstance(stmt, ast.If) and not stmt.orelse and len(stmt.body) == 1:
            inner = stmt.body[0]
            if isinstance(inner, ast.Return) and (
                inner.value is None
                or (isinstance(inner.value, ast.Constant) and inner.value.value is None)
            ):
                out.append(("return None", stmt))
            elif isinstance(inner, (ast.Raise,)):
                out.append(("raise", stmt))
            elif isinstance(inner, ast.Continue) or isinstance(inner, ast.Break):
                out.append(("exit loop", stmt))
    return out


def _e1(ctx):
    guards = _guard_conditions(ctx)
    if not guards:
        return []
    what, stmt = guards[0]
    src = ctx.seg(stmt.test)
    if not ctx.span_ok(src) or len(src) > 160:
        return []
    return [Fact("E1", {"intent": "E1", "exit_kind": what}, src, "substring")]


register("E1", "E/F", description="condition for early return None/exit")(
    (lambda ctx: bool(_guard_conditions(ctx)), _e1)
)


def _has_unreachable(ctx):
    def block_dead(stmts):
        for i, s in enumerate(stmts[:-1]):
            if isinstance(s, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                return True
        return False

    for n in ctx.body_walk():
        if _in_nested_fn(ctx.fn, n):
            continue
        for attr in ("body", "orelse", "finalbody"):
            blk = getattr(n, attr, None)
            if isinstance(blk, list) and blk and block_dead(blk):
                return True
        if isinstance(n, (ast.If, ast.While)) and isinstance(n.test, ast.Constant) \
                and n.test.value is False:
            return True
    return False


def _e2(ctx):
    return [Fact("E2", {"intent": "E2"},
                 "yes" if _has_unreachable(ctx) else "no", "yesno")]


register("E2", "E/F", description="unreachable code present?")(
    (lambda ctx: True, _e2)
)


def _e3(ctx):
    rets = ctx.returns_with_value()
    srcs = []
    for r in rets:
        s = ctx.seg(r.value)
        if not s or len(s) > 80:
            return []
        if s not in srcs:
            srcs.append(s)
    if not (2 <= len(srcs) <= 4):
        return []
    return [Fact("E3", {"intent": "E3"}, "; ".join(srcs), "list")]


register("E3", "E/F", description="distinct return expressions")(
    (lambda ctx: len(ctx.returns_with_value()) >= 2, _e3)
)


def _f1(ctx):
    assigned = assigned_names(ctx)
    pset = set(ctx.params())
    once = sorted(v for v, ls in assigned.items()
                  if len(ls) == 1 and v not in pset and not v.startswith("_"))
    if not once:
        return []
    var = ctx.rng.choice(once)
    return [Fact("F1", {"intent": "F1", "var": var},
                 f"L{assigned[var][0]}", "line_ref")]


register("F1", "E/F", description="where does V get assigned")(
    (lambda ctx: any(
        len(ls) == 1 and v not in set(ctx.params()) and not v.startswith("_")
        for v, ls in assigned_names(ctx).items()
    ), _f1)
)


def _f2(ctx):
    """Static backward slice from return expressions to params. HELD OUT."""
    deps: dict[str, set[str]] = {}
    for n in ctx.body_walk():
        if _in_nested_fn(ctx.fn, n):
            continue
        tgt_names, src_node = [], None
        if isinstance(n, ast.Assign):
            src_node = n.value
            for t in n.targets:
                tgt_names += [l.id for l in ast.walk(t) if isinstance(l, ast.Name)]
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            src_node = n.value
            tgt_names = [n.target.id]
            deps.setdefault(n.target.id, set()).add(n.target.id)
        elif isinstance(n, ast.AnnAssign) and n.value and isinstance(n.target, ast.Name):
            src_node = n.value
            tgt_names = [n.target.id]
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            src_node = n.iter
            tgt_names = [l.id for l in ast.walk(n.target) if isinstance(l, ast.Name)]
        if src_node is not None:
            srcs = {l.id for l in ast.walk(src_node)
                    if isinstance(l, ast.Name) and isinstance(l.ctx, ast.Load)}
            for t in tgt_names:
                deps.setdefault(t, set()).update(srcs)
    start: set[str] = set()
    for r in ctx.returns_with_value():
        start.update(l.id for l in ast.walk(r.value)
                     if isinstance(l, ast.Name) and isinstance(l.ctx, ast.Load))
    seen = set(start)
    frontier = list(start)
    while frontier:
        v = frontier.pop()
        for d in deps.get(v, ()):
            if d not in seen:
                seen.add(d)
                frontier.append(d)
    influencing = [p for p in ctx.params() if p in seen and p not in ("self", "cls")]
    ans = ", ".join(influencing) if influencing else "none"
    return [Fact("F2", {"intent": "F2"}, ans, "list", held_out=True)]


register("F2", "E/F", held_out=True,
         description="params influencing return value (static slice) [HELD OUT]")(
    (lambda ctx: bool(ctx.returns_with_value()), _f2)
)


def _unused(ctx):
    loaded = loaded_names(ctx)
    assigned = assigned_names(ctx)
    unused_params = [p for p in ctx.params()
                     if p not in loaded and p not in ("self", "cls")
                     and not p.startswith("_")]
    unused_locals = [v for v in assigned
                     if v not in loaded and not v.startswith("_")
                     and v not in set(ctx.params())]
    return unused_params, unused_locals


def _f3(ctx):
    up, ul = _unused(ctx)
    return [Fact("F3", {"intent": "F3"}, "yes" if (up or ul) else "no", "yesno")]


register("F3", "E/F", description="unused variables/params present?")(
    (lambda ctx: True, _f3)
)


def _f4(ctx):
    muts = mutated_params(ctx)
    muts = [m for m in muts if m not in ("self", "cls")]
    ans = ", ".join(muts) if muts else "none"
    return [Fact("F4", {"intent": "F4"}, ans, "list")]


register("F4", "E/F", description="which args are mutated (AST evidence)")(
    (lambda ctx: len([p for p in ctx.params() if p not in ("self", "cls")]) > 0, _f4)
)


# ==========================================================================
# GROUP G — quality (4 intents)
# ==========================================================================


def _g1(ctx):
    a = ctx.fn.args
    defaults = list(a.defaults) + [d for d in a.kw_defaults if d is not None]
    mutable = any(isinstance(d, (ast.List, ast.Dict, ast.Set, ast.SetComp,
                                 ast.ListComp, ast.DictComp)) for d in defaults)
    return [Fact("G1", {"intent": "G1"}, "yes" if mutable else "no", "yesno")]


register("G1", "G", description="mutable default param present?")(
    (lambda ctx: bool(ctx.fn.args.defaults or
                      [d for d in ctx.fn.args.kw_defaults if d is not None]), _g1)
)


def _g2(ctx):
    bare = any(isinstance(n, ast.ExceptHandler) and n.type is None
               for n in ctx.body_walk())
    return [Fact("G2", {"intent": "G2"}, "yes" if bare else "no", "yesno")]


register("G2", "G", description="bare except present?")(
    (lambda ctx: any(isinstance(n, ast.Try) for n in ctx.body_walk()), _g2)
)

_SHADOWABLE = frozenset(
    "list dict set str int float bool type id input filter map sum min max len "
    "next iter range object bytes tuple hash dir vars format all any zip".split()
)


def _shadowed(ctx):
    names = set(assigned_names(ctx)) | set(ctx.params())
    return sorted(names & _SHADOWABLE)


def _g3(ctx):
    sh = _shadowed(ctx)
    facts = [Fact("G3", {"intent": "G3"}, "yes" if sh else "no", "yesno")]
    if sh:
        facts.append(Fact("G3", {"intent": "G3", "aspect": "which"},
                          ctx.rng.choice(sh), "name"))
    return facts


register("G3", "G", description="shadowed builtin used?")((lambda ctx: True, _g3))


def _g4(ctx):
    decs = {ctx.seg(d) for d in ctx.fn.decorator_list}
    if "staticmethod" in decs:
        return []
    expected = "cls" if "classmethod" in decs else "self"
    a = ctx.fn.args
    pos = a.posonlyargs + a.args
    first = pos[0].arg if pos else None
    anomalous = first != expected
    return [Fact("G4", {"intent": "G4"}, "yes" if anomalous else "no", "yesno")]


register("G4", "G", description="first-param-not-self in method?")(
    (lambda ctx: bool(ctx.row["is_method"]), _g4)
)


# ==========================================================================
# GROUP H — docs (3 intents)
# ==========================================================================


def _h1(ctx):
    node = ctx.docstring_node()
    doc = node.value
    if not (0 < len(doc) <= 300):
        return []
    if not ctx.span_ok(doc):
        return []  # escape sequences differ from source; drop
    return [Fact("H1", {"intent": "H1"}, doc, "substring")]


register("H1", "H", description="recall docstring verbatim")(
    (lambda ctx: ctx.docstring_node() is not None, _h1)
)

_RAISES_SEC = re.compile(
    r"(?:^|\n)\s*(?:Raises?\s*[:\n]|:raises?\s+)", re.IGNORECASE
)
_RAISES_NAMES = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Warning|Interrupt|Exit|Iteration))\b")


def _doc_raises(ctx):
    node = ctx.docstring_node()
    if node is None:
        return None
    doc = node.value
    m = _RAISES_SEC.search(doc)
    if not m:
        return None
    section = doc[m.start():]
    return sorted(set(_RAISES_NAMES.findall(section)))


def _h2(ctx):
    """Docstring Raises section vs actual explicit raises. The catalog's
    injected-mismatch variant (mutating docstrings to create mismatches) is a
    corpus-modification step that belongs to the phrasing stage; here the
    natural match/mismatch is emitted. spec_question."""
    documented = _doc_raises(ctx)
    if documented is None:
        return []
    actual = sorted(set(explicit_raises(ctx)))
    match = set(documented) == set(actual) and bool(actual)
    return [Fact("H2", {"intent": "H2"}, "yes" if match else "no", "yesno",
                 spec_question=True)]


register("H2", "H", description="docstring Raises matches actual raises?")(
    (lambda ctx: _doc_raises(ctx) is not None, _h2)
)

_DOC_STYLES = ["Google style", "NumPy style", "Sphinx/reST style", "plain prose"]


def _doc_style(doc: str) -> Optional[str]:
    numpy = re.search(r"\n\s*(Parameters|Returns|Raises)\s*\n\s*-{3,}", doc)
    google = re.search(r"\n\s*(Args|Arguments|Returns|Raises|Yields)\s*:\s*\n", doc)
    sphinx = re.search(r":(param|returns?|rtype|raises)\b", doc)
    hits = [s for s, m in
            zip(["NumPy style", "Google style", "Sphinx/reST style"],
                [numpy, google, sphinx]) if m]
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        return "plain prose"
    return None  # ambiguous


def _h3(ctx):
    node = ctx.docstring_node()
    facts = [Fact("H3", {"intent": "H3", "aspect": "present"},
                  "yes" if node is not None else "no", "yesno")]
    if node is not None and len(node.value) > 30:
        style = _doc_style(node.value)
        if style is not None:
            distractors = [s for s in _DOC_STYLES if s != style]
            facts += _mcq(ctx, "H3", style, distractors,
                          extra_seed={"aspect": "style"})
    return facts


register("H3", "H", description="docstring present? + style (MCQ)")(
    (lambda ctx: True, _h3)
)


# ==========================================================================
# GROUP I — descriptive retrieval over STACKS (5 intents)
# ==========================================================================
# These operate over a stack of N functions. The description text is
# 8B-phrased later; seeds carry describe_row_idx and needs_phrasing=True.
# Sibling/stack construction is deterministic (see qa_coverage.build_stacks).

STACK_SIZE = 8


def _stack_rng(stack):
    key = "|".join(r["qualified"] for r in stack["rows"])
    return random.Random(int(hashlib.md5(key.encode()).hexdigest()[:12], 16))


def extract_stack_intents(stack: dict, decoy: Optional[dict]) -> list[tuple[dict, Fact]]:
    """stack = {"rows": [row dicts], "row_idxs": [full-split idxs]}.
    decoy = one row from a DIFFERENT stack (for I2/I5-no).
    Returns [(target_row, Fact)] so triples carry the right row_idx."""
    rng = _stack_rng(stack)
    rows, idxs = stack["rows"], stack["row_idxs"]
    names = [r["func_name"].split(".")[-1] for r in rows]
    out = []
    ti = rng.randrange(len(rows))
    target, tidx = rows[ti], idxs[ti]
    sibling_idxs = [i for i in idxs if i != tidx]

    # I1: which function does <description>? -> name
    out.append((target, Fact(
        "I1",
        {"intent": "I1", "describe_row_idx": tidx, "stack_row_idxs": idxs},
        names[ti], "name", needs_phrasing=True)))

    # I2: absence — does any function here do X? -> no (decoy from elsewhere)
    if decoy is not None and decoy["func_name"].split(".")[-1] not in names:
        out.append((rows[0], Fact(
            "I2",
            {"intent": "I2", "describe_row_idx": decoy["row_idx"],
             "stack_row_idxs": idxs},
            "no", "yesno", needs_phrasing=True)))

    # I3: which-of-N MCQ (options = 4 function names from the stack)
    t2 = rng.randrange(len(rows))
    others = [n for j, n in enumerate(names) if j != t2 and n != names[t2]]
    rng.shuffle(others)
    opts = [names[t2]] + others[:3]
    if len(set(opts)) == 4:
        rng.shuffle(opts)
        letter = "ABCD"[opts.index(names[t2])]
        out.append((rows[t2], Fact(
            "I3",
            {"intent": "I3", "describe_row_idx": idxs[t2],
             "stack_row_idxs": idxs, "options": opts},
            letter, "letter", mcq=True, needs_phrasing=True)))

    # I4: composed — "the one that <description>: <A-group question>"
    inner = _i4_inner(target, rng)
    if inner is not None:
        inner_seed, inner_answer, inner_type = inner
        out.append((target, Fact(
            "I4",
            {"intent": "I4", "describe_row_idx": tidx,
             "stack_row_idxs": idxs, "inner": inner_seed},
            inner_answer, inner_type, needs_phrasing=True)))

    # I5: reverse — does <name> match <description>? (hash parity picks yes/no)
    yes_case = rng.random() < 0.5
    if yes_case:
        out.append((target, Fact(
            "I5",
            {"intent": "I5", "name": names[ti], "describe_row_idx": tidx,
             "stack_row_idxs": idxs},
            "yes", "yesno", needs_phrasing=True)))
    elif decoy is not None and decoy["func_name"].split(".")[-1] != names[ti]:
        out.append((target, Fact(
            "I5",
            {"intent": "I5", "name": names[ti],
             "describe_row_idx": decoy["row_idx"], "stack_row_idxs": idxs},
            "no", "yesno", needs_phrasing=True)))
    return out


def _i4_inner(row, rng):
    """A-group fact about the described function (param count / is-async /
    param names). Uses a fresh ctx; returns (inner_seed, answer, type)."""
    try:
        ctx = make_ctx(row, row.get("row_idx", -1))
    except (ParseFailure, SyntaxError, RecursionError):
        return None
    choices = []
    choices.append(({"intent": "A2"}, str(len(ctx.params())), "number"))
    choices.append(({"intent": "A9"},
                    "yes" if isinstance(ctx.fn, ast.AsyncFunctionDef) else "no",
                    "yesno"))
    if ctx.params():
        choices.append(({"intent": "A3"}, ", ".join(ctx.params()), "list"))
    return rng.choice(choices)


# register stack intents for coverage bookkeeping (extract goes via
# extract_stack_intents, not per-fn)
for _iid, _desc in [
    ("I1", "which function does <description>? -> name"),
    ("I2", "absence: does any function here do X? -> no"),
    ("I3", "which-of-N MCQ variant"),
    ("I4", "composed description + A-question"),
    ("I5", "reverse: does <name> match <description>?"),
]:
    REGISTRY[_iid] = IntentSpec(_iid, "I", "stack", lambda ctx: False,
                                lambda ctx: [], description=_desc)


# --------------------------------------------------------------------------
# Per-function driver
# --------------------------------------------------------------------------

FN_INTENT_IDS = [iid for iid, spec in REGISTRY.items() if spec.kind == "fn"]


def extract_row(ctx: FnCtx) -> tuple[list[Fact], dict]:
    """Run all per-fn intents. Returns (facts, stats) where stats counts
    applicability and drops."""
    facts = []
    stats = {"applicable": [], "leak_dropped": 0, "span_dropped": 0}
    for iid in FN_INTENT_IDS:
        spec = REGISTRY[iid]
        try:
            if not spec.applicable(ctx):
                continue
        except (RecursionError, Exception):
            continue
        stats["applicable"].append(iid)
        try:
            fs = spec.extract(ctx)
        except RecursionError:
            continue
        except Exception:
            continue
        for f in fs:
            if not leak_check(f):
                stats["leak_dropped"] += 1
                continue
            facts.append(f)
    return facts, stats


def fact_to_triple(fact: Fact, row: dict, row_idx: int) -> dict:
    return {
        "row_idx": row_idx,
        "qualified": row["qualified"],
        "intent": fact.intent,
        "group": REGISTRY[fact.intent].group,
        "held_out": fact.held_out,
        "needs_luna": fact.needs_luna,
        "needs_phrasing": fact.needs_phrasing,
        "question_seed": fact.question_seed,
        "answer": fact.answer,
        "answer_type": fact.answer_type,
    }
