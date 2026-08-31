"""Execute runnable QA intents in an isolated subprocess.

The pipeline applies an AST self-containedness screen, synthesizes deterministic
inputs from annotations, defaults, and usage, executes the function in a
sandbox, and derives catalog answers from results and line traces.

Each batch runs in a separate ``python -I`` child process with:

- restricted builtins and guarded imports over an allowlist,
- pre-injected allowlisted standard-library modules,
- CPU and address-space limits where supported,
- a two-second per-call timer and a 200,000-event trace cap, and
- a parent-side hard timeout and process termination.

The static screen rejects imports outside the allowlist, unresolved free
variables, unsafe attribute roots, dunder attributes, I/O-shaped names,
asynchronous functions, generators, and global state. Process isolation is
defense in depth. A failed batch is retried row by row so one hostile example
cannot discard unrelated rows.

Determinism: input synthesis is seeded by md5(qualified); every input is
executed twice and runs whose result/exception reprs differ are dropped
(status "nondet"). random/time/os cannot pass the screen in the first place.

Line numbers are 1-based relative to the code snippet, matching qa_extract.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import subprocess
import sys
from typing import Optional

try:
    from qa_extract import Fact, IntentSpec, REGISTRY
except ImportError:  # sandbox child runs `python -I` (no script dir on path)
    Fact = IntentSpec = REGISTRY = None

# --------------------------------------------------------------------------
# 1) Self-containedness screen
# --------------------------------------------------------------------------

# Whitelist per spec: math itertools functools collections string re json.
# Additions (justified): heapq + bisect — pure in-memory algorithmic stdlib,
# no I/O, no ambient authority, common in this corpus's algorithmic functions.
ALLOWED_MODULES = frozenset(
    "math itertools functools collections string re json heapq bisect".split()
)

SAFE_BUILTIN_NAMES = frozenset(
    "abs all any ascii bin bool bytearray bytes callable chr complex dict "
    "divmod enumerate filter float format frozenset hash hex int isinstance "
    "issubclass iter len list map max min next object oct ord pow print range "
    "repr reversed round set slice sorted str sum tuple zip "
    "True False None NotImplemented Ellipsis "
    "Exception BaseException ValueError TypeError KeyError IndexError "
    "AttributeError ZeroDivisionError ArithmeticError OverflowError "
    "RuntimeError StopIteration NotImplementedError AssertionError "
    "LookupError NameError UnicodeDecodeError UnicodeEncodeError".split()
)

# names whose mere use fails the screen (I/O, reflection, escape hatches)
FORBIDDEN_NAMES = frozenset(
    "open input eval exec compile __import__ globals locals vars getattr "
    "setattr delattr exit quit breakpoint super memoryview type id object "
    "help copyright credits license classmethod staticmethod property".split()
) - {"object", "id"}  # id/object are harmless


def _binds(node) -> set[str]:
    """Names bound within a function node (params, assignments, loops, withs,
    excepts, comprehensions, imports, nested defs)."""
    bound = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
            if n is not node and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = n.args
                for p in a.posonlyargs + a.args + a.kwonlyargs:
                    bound.add(p.arg)
                if a.vararg:
                    bound.add(a.vararg.arg)
                if a.kwarg:
                    bound.add(a.kwarg.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                bound.add(a.asname or a.name)
        elif isinstance(n, ast.Lambda):
            a = n.args
            for p in a.posonlyargs + a.args + a.kwonlyargs:
                bound.add(p.arg)
            if a.vararg:
                bound.add(a.vararg.arg)
            if a.kwarg:
                bound.add(a.kwarg.arg)
    a = node.args
    for p in a.posonlyargs + a.args + a.kwonlyargs:
        bound.add(p.arg)
    if a.vararg:
        bound.add(a.vararg.arg)
    if a.kwarg:
        bound.add(a.kwarg.arg)
    return bound


def _attr_root(node):
    """Descend an Attribute/Subscript/Call chain to its root expression."""
    while True:
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return node


def screen(fn: ast.AST, row: dict) -> tuple[bool, str, dict]:
    """Returns (ok, reason, meta). meta: seq_params (for C2), max_tokens."""
    if isinstance(fn, ast.AsyncFunctionDef):
        return False, "async", {}
    if row.get("n_tokens", 0) > 600:
        return False, "too_large", {}
    if fn.decorator_list:
        return False, "decorated", {}
    bound = _binds(fn)
    params = set()
    a = fn.args
    for p in a.posonlyargs + a.args + a.kwonlyargs:
        params.add(p.arg)
    if a.vararg:
        params.add(a.vararg.arg)
    if a.kwarg:
        params.add(a.kwarg.arg)
    called_params = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await)):
            return False, "generator_or_await", {}
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            return False, "global_nonlocal", {}
        if isinstance(n, ast.Import):
            for al in n.names:
                if al.name.split(".")[0] not in ALLOWED_MODULES:
                    return False, f"import:{al.name}", {}
        if isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] not in ALLOWED_MODULES:
                return False, f"import:{n.module}", {}
        if isinstance(n, ast.Attribute):
            if n.attr.startswith("__"):
                return False, "dunder_attr", {}
            root = _attr_root(n.value)
            if isinstance(root, ast.Name):
                if root.id not in bound and root.id not in ALLOWED_MODULES:
                    return False, f"attr_on_global:{root.id}", {}
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id in FORBIDDEN_NAMES:
                return False, f"forbidden:{n.id}", {}
            if n.id not in bound and n.id not in SAFE_BUILTIN_NAMES \
                    and n.id not in ALLOWED_MODULES:
                return False, f"free_var:{n.id}", {}
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id in params:
            called_params.add(n.func.id)
    if called_params:
        return False, "param_called", {}
    if ("self" in params or "cls" in params):
        used = {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        if "self" in used or "cls" in used:
            return False, "uses_self", {}
    return True, "", {}


# --------------------------------------------------------------------------
# 2) Input synthesis (deterministic, seeded by qualified)
# --------------------------------------------------------------------------

_POOLS = {
    "int": [0, 2, 3, 7, -1],
    "float": [0.5, 2.0, -1.5],
    "str": ["ab", "banana", "x y z"],
    "list": [[1, 2, 3], [3, 1, 2, 1], []],
    "list_str": [["a", "bb", "c"], []],
    "dict": [{"a": 1, "b": 2}, {}],
    "bool": [True, False],
    # tuples ride a JSON protocol to the sandbox child and would silently
    # become lists; use lists outright so seed reprs stay honest
    "tuple": [[1, 2], []],
    "any": [3, "ab", [1, 2, 3], 0],
}

_EMPTY = {"str": "", "list": [], "list_str": [], "dict": {}, "tuple": [],
          "any": [], "int": 0, "float": 0.0, "bool": False}


def _json_safe(v, depth=0):
    """Synthesized values must survive the JSON protocol unchanged.
    Returns (ok, converted): tuples -> lists; Ellipsis/bytes/sets/objects
    rejected; dict keys must be str."""
    if depth > 4:
        return False, None
    if v is None or isinstance(v, (bool, int, float, str)):
        return True, v
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            ok, c = _json_safe(x, depth + 1)
            if not ok:
                return False, None
            out.append(c)
        return True, out
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str):
                return False, None
            ok, c = _json_safe(x, depth + 1)
            if not ok:
                return False, None
            out[k] = c
        return True, out
    return False, None


def _param_kind(name: str, ann: Optional[ast.AST], default, usage: dict) -> tuple[str, bool]:
    """Returns (kind, grounded). grounded=True iff the kind comes from the
    function's OWN type expectations (annotation, default value, or usage
    evidence). Name heuristics and the 'any' fallback are synthesizer guesses,
    not expectations -> grounded=False. D7 (exception prediction) only emits
    from runs whose every param kind is grounded, so that raises reflect
    behavior on validly-typed inputs rather than wrong-type artifacts."""
    ann_src = ast.unparse(ann).lower() if ann is not None else ""
    for key, kind in [("list[str]", "list_str"), ("sequence[str]", "list_str"),
                      ("list", "list"), ("sequence", "list"), ("iterable", "list"),
                      ("tuple", "tuple"), ("dict", "dict"), ("mapping", "dict"),
                      ("str", "str"), ("bool", "bool"), ("float", "float"),
                      ("int", "int")]:
        if key in ann_src:
            return kind, True
    if default is not None:
        for typ, kind in [(bool, "bool"), (int, "int"), (float, "float"),
                          (str, "str"), (list, "list"), (dict, "dict"),
                          (tuple, "tuple")]:
            if isinstance(default, typ):
                return kind, True
    if usage.get("iterated") or usage.get("len"):
        return "list", True
    if usage.get("str_method"):
        return "str", True
    if usage.get("dict_method"):
        return "dict", True
    if usage.get("arith"):
        return "int", True
    nl = name.lower()
    if any(t in nl for t in ("count", "num", "idx", "index", "size", "n_", "len")) or nl == "n":
        return "int", False
    if any(t in nl for t in ("name", "text", "word", "path", "key", "sep", "prefix", "suffix", "char")):
        return "str", False
    if any(t in nl for t in ("items", "values", "nums", "arr", "lst", "list", "seq", "data", "elements")):
        return "list", False
    if any(t in nl for t in ("flag", "enable", "verbose", "is_", "has_")):
        return "bool", False
    return "any", False


_STR_METHODS = frozenset("lower upper strip split join startswith endswith "
                         "replace format encode title capitalize".split())
_DICT_METHODS = frozenset("items keys values get setdefault".split())


def _usage_of(fn: ast.AST) -> dict[str, dict]:
    usage: dict[str, dict] = {}

    def u(name):
        return usage.setdefault(name, {})

    for n in ast.walk(fn):
        if isinstance(n, (ast.For, ast.AsyncFor)) and isinstance(n.iter, ast.Name):
            u(n.iter.id)["iterated"] = True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id == "len" and n.args and isinstance(n.args[0], ast.Name):
            u(n.args[0].id)["len"] = True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and isinstance(n.func.value, ast.Name):
            if n.func.attr in _STR_METHODS:
                u(n.func.value.id)["str_method"] = True
            if n.func.attr in _DICT_METHODS:
                u(n.func.value.id)["dict_method"] = True
        if isinstance(n, ast.BinOp):
            for side in (n.left, n.right):
                if isinstance(side, ast.Name):
                    u(side.id)["arith"] = True
    return usage


def _literal_default(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def synth_inputs(fn: ast.AST, qualified: str, max_combos: int = 6):
    """Returns (combos, empty_combo_param). combo = {"args": [...],
    "kwargs": {...}}. Values are plain literals (JSON-serializable)."""
    rng = random.Random(int(hashlib.md5(("inp|" + qualified).encode())
                            .hexdigest()[:12], 16))
    a = fn.args
    usage = _usage_of(fn)
    pos_params = a.posonlyargs + a.args
    pos_defaults = dict(zip([p.arg for p in pos_params[len(pos_params) - len(a.defaults):]],
                            a.defaults))
    plans = []  # (name, kind, candidates, positional?)
    typed_ok = True  # every synthesized param kind grounded in the fn's own
    # type expectations (annotation/default/usage) -> raises are D7-eligible
    for p in pos_params:
        if p.arg in ("self", "cls"):
            plans.append((p.arg, "none", [None], True))
            continue
        d = _literal_default(pos_defaults[p.arg]) if p.arg in pos_defaults else None
        d_ok, d = _json_safe(d) if d is not None else (False, None)
        kind, grounded = _param_kind(p.arg, p.annotation, d, usage.get(p.arg, {}))
        typed_ok = typed_ok and grounded
        cands = list(_POOLS.get(kind, _POOLS["any"]))
        if d_ok and d is not None and d not in cands:
            cands.insert(1, d)
        plans.append((p.arg, kind, cands, True))
    kw_plans = []
    for p, d_node in zip(a.kwonlyargs, a.kw_defaults):
        if d_node is not None:
            continue  # has default: omit
        d = None
        kind, grounded = _param_kind(p.arg, p.annotation, d, usage.get(p.arg, {}))
        typed_ok = typed_ok and grounded
        kw_plans.append((p.arg, kind, list(_POOLS.get(kind, _POOLS["any"]))))

    def build(pick):
        args = [pick(name, kind, cands) for name, kind, cands, _ in plans]
        kwargs = {name: pick(name, kind, cands) for name, kind, cands in kw_plans}
        return {"args": args, "kwargs": kwargs}

    combos = [build(lambda n, k, c: c[0])]
    for _ in range(max_combos * 2):
        cb = build(lambda n, k, c: rng.choice(c))
        if cb not in combos:
            combos.append(cb)
        if len(combos) >= max_combos:
            break
    # empty-situation combo for C2 (first sequence-ish non-self param)
    empty_param = None
    for name, kind, cands, _ in plans:
        if kind in ("list", "list_str", "str", "dict", "tuple"):
            empty_param = name
            break
    if empty_param is not None:
        def pick_empty(n, k, c):
            return _EMPTY.get(k, []) if n == empty_param else c[0]
        cb = build(pick_empty)
        cb["empty_param"] = empty_param
        combos.append(cb)
    if typed_ok:
        for cb in combos:
            cb["typed_ok"] = True
    return combos, empty_param


# --------------------------------------------------------------------------
# 3) Sandbox child (this file run with --sandbox-child)
# --------------------------------------------------------------------------

CHILD_FLAG = "--sandbox-child"
_TRACE_CAP = 200_000
_REPR_CAP = 120


def _canonical_repr(v, depth=0):
    """repr for gradeable answers; None if the value isn't canonical
    (objects, sets whose repr order could drift, NaN ambiguity is fine)."""
    if depth > 4:
        return None
    if v is None or isinstance(v, (bool, int)):
        return repr(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (str, bytes)):
        return repr(v) if len(repr(v)) <= _REPR_CAP else None
    if isinstance(v, (list, tuple)):
        parts = [_canonical_repr(x, depth + 1) for x in v]
        if any(p is None for p in parts):
            return None
        inner = ", ".join(parts)
        if isinstance(v, tuple):
            r = "(" + inner + ("," if len(v) == 1 else "") + ")"
        else:
            r = "[" + inner + "]"
        return r if len(r) <= _REPR_CAP else None
    if isinstance(v, dict):
        parts = []
        for k, x in v.items():
            kr, xr = _canonical_repr(k, depth + 1), _canonical_repr(x, depth + 1)
            if kr is None or xr is None:
                return None
            parts.append(f"{kr}: {xr}")
        r = "{" + ", ".join(parts) + "}"
        return r if len(r) <= _REPR_CAP else None
    return None


def _child_main():
    import io
    import resource
    import signal

    try:
        resource.setrlimit(resource.RLIMIT_CPU, (10, 12))
    except (ValueError, OSError):
        pass
    try:  # not enforced on darwin; real backstop is alarm + parent kill
        resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    except (ValueError, OSError):
        pass

    import builtins as B

    def guarded_import(name, *args, **kw):
        if name.split(".")[0] not in ALLOWED_MODULES:
            raise ImportError(f"blocked: {name}")
        return B.__import__(name, *args, **kw)

    safe_builtins = {n: getattr(B, n) for n in SAFE_BUILTIN_NAMES if hasattr(B, n)}
    safe_builtins["__import__"] = guarded_import
    safe_builtins["__build_class__"] = B.__build_class__
    safe_builtins["__name__"] = "qa_sandbox"
    premods = {m: __import__(m) for m in ALLOWED_MODULES}

    class _Timeout(Exception):
        pass

    def _alarm(sig, frm):
        raise _Timeout()

    signal.signal(signal.SIGALRM, _alarm)

    tasks = json.load(sys.stdin)
    results = []
    for task in tasks:
        results.append(_child_run_one(task, safe_builtins, premods, signal, _Timeout, io))
    json.dump(results, sys.stdout)


def _child_run_one(task, safe_builtins, premods, signal, _Timeout, io):
    out = {"qualified": task["qualified"], "status": "ok", "runs": []}
    try:
        code_obj = compile(task["code"], "<qa_fn>", "exec")
    except SyntaxError:
        out["status"] = "compile_error"
        return out
    ns = {"__builtins__": safe_builtins, **premods}
    try:
        signal.setitimer(signal.ITIMER_REAL, 2.0)
        exec(code_obj, ns)
        signal.setitimer(signal.ITIMER_REAL, 0)
    except BaseException:
        signal.setitimer(signal.ITIMER_REAL, 0)
        out["status"] = "def_error"
        return out
    fn = ns.get(task["fn_name"])
    if not callable(fn):
        out["status"] = "def_error"
        return out

    for combo in task["combos"]:
        rec = {"args": [repr(v) for v in combo["args"]],
               "kwargs": {k: repr(v) for k, v in combo["kwargs"].items()}}
        if "empty_param" in combo:
            rec["empty_param"] = combo["empty_param"]
        if combo.get("typed_ok"):
            rec["typed_ok"] = True
        r1 = _one_call(fn, combo, signal, _Timeout, io, trace=False)
        r2 = _one_call(fn, combo, signal, _Timeout, io, trace=True)
        if r1["outcome"] != r2["outcome"]:
            rec["status"] = "nondet"
        else:
            rec.update(r2)
            rec["status"] = r2["kind"]
        rec.pop("kind", None)
        rec.pop("outcome", None)
        results_ok = rec.get("status") in ("ok", "exception")
        out["runs"].append(rec)
        if not results_ok and rec["status"] == "timeout":
            break  # this fn is slow; don't burn more combos
    return out


def _one_call(fn, combo, signal, _Timeout, io, trace):
    import copy
    import sys as _sys

    counts: dict[int, int] = {}
    snaps: dict[int, dict] = {}
    n_events = [0]
    prev_ln = [None]

    def snapshot(frame):
        if prev_ln[0] is None:
            return
        ln = prev_ln[0]
        if ln not in snaps:
            d = {}
            for k, v in frame.f_locals.items():
                if k.startswith("__"):
                    continue
                r = _canonical_repr(v)
                if r is not None:
                    d[k] = r
                if len(d) >= 10:
                    break
            snaps[ln] = d
        prev_ln[0] = None

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != "<qa_fn>":
            return None
        n_events[0] += 1
        if n_events[0] > _TRACE_CAP:
            raise _Timeout()
        if event == "line":
            snapshot(frame)
            counts[frame.f_lineno] = counts.get(frame.f_lineno, 0) + 1
            prev_ln[0] = frame.f_lineno
        elif event == "return":
            snapshot(frame)
        return tracer

    try:
        args = copy.deepcopy(combo["args"])
        kwargs = copy.deepcopy(combo["kwargs"])
    except Exception:
        return {"kind": "synth_error", "outcome": "synth_error"}
    old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        signal.setitimer(signal.ITIMER_REAL, 2.0)
        if trace:
            _sys.settrace(tracer)
        try:
            result = fn(*args, **kwargs)
            rrepr = _canonical_repr(result)
            rec = {"kind": "ok", "result": rrepr,
                   "outcome": ("ok", rrepr)}
        except _Timeout:
            rec = {"kind": "timeout", "outcome": "timeout"}
        except RecursionError:
            rec = {"kind": "exception", "exc": "RecursionError",
                   "outcome": ("exc", "RecursionError")}
        except BaseException as e:
            rec = {"kind": "exception", "exc": type(e).__name__,
                   "outcome": ("exc", type(e).__name__)}
    finally:
        _sys.settrace(None)
        signal.setitimer(signal.ITIMER_REAL, 0)
        _sys.stdout = old_stdout
    if trace:
        rec["line_counts"] = {str(k): v for k, v in counts.items()}
        rec["snapshots"] = snaps
    return rec


# --------------------------------------------------------------------------
# Parent side: batch runner + fact building (D1-D7, C2-exec)
# --------------------------------------------------------------------------


def run_sandbox_batch(tasks: list[dict], timeout: Optional[float] = None) -> list[dict]:
    """tasks: [{qualified, code, fn_name, combos}]. Returns child records;
    on batch crash/timeout, retries rows individually so one bad row can only
    sink itself."""
    if not tasks:
        return []
    if timeout is None:
        timeout = 20 + 3.0 * len(tasks)
    payload = json.dumps(tasks)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", __file__, CHILD_FLAG],
            input=payload, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode == 0 and proc.stdout:
            return json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    if len(tasks) == 1:
        return [{"qualified": tasks[0]["qualified"], "status": "crashed", "runs": []}]
    mid = len(tasks) // 2
    return run_sandbox_batch(tasks[:mid]) + run_sandbox_batch(tasks[mid:])


def _fmt_input(rec):
    parts = list(rec["args"]) + [f"{k}={v}" for k, v in rec["kwargs"].items()]
    return ", ".join(parts)


def build_exec_facts(record: dict, fn: ast.AST, qualified: str) -> list[Fact]:
    """Turn a child record into D1-D7 + C2 facts. Deterministic (rng seeded
    by qualified)."""
    if record["status"] != "ok":
        return []
    rng = random.Random(int(hashlib.md5(("fact|" + qualified).encode())
                            .hexdigest()[:12], 16))
    runs = record["runs"]
    ok_runs = [r for r in runs if r.get("status") == "ok" and r.get("result")]
    exc_runs = [r for r in runs if r.get("status") == "exception"]
    facts = []

    # D1: output of f(input). Skip runs whose output already appears verbatim
    # in the input string (None/True/echoed values would leak into the
    # question; leak_check exempts such trivial tokens, so filter here).
    d1_runs = [r for r in ok_runs if r["result"] not in _fmt_input(r)]
    if d1_runs:
        r = d1_runs[0]
        facts.append(Fact("D1", {"intent": "D1", "input": _fmt_input(r)},
                          r["result"], "text"))

    # D2: give an input producing output Y (Y in seed BY DEFINITION;
    # answer = the input). Only when Y is unique across tried inputs.
    seen: dict[str, int] = {}
    for r in ok_runs:
        seen[r["result"]] = seen.get(r["result"], 0) + 1
    uniq = [r for r in ok_runs if seen[r["result"]] == 1
            and r["result"] != _fmt_input(r)]
    if uniq:
        r = uniq[0]
        facts.append(Fact("D2", {"intent": "D2", "output": r["result"]},
                          _fmt_input(r), "text", leak_by_design=True))

    # trace-based facts use the first ok run with a trace
    traced = next((r for r in ok_runs if r.get("line_counts")), None)
    if traced is not None:
        counts = {int(k): v for k, v in traced["line_counts"].items()}
        snaps = {int(k): v for k, v in traced.get("snapshots", {}).items()}
        inp = _fmt_input(traced)

        # D3: value of var V after line N (first execution of N)
        cand = [(ln, var, val) for ln, d in sorted(snaps.items())
                for var, val in sorted(d.items())
                if not var.startswith("_") and len(val) <= 80
                and val not in inp]  # answer-in-input leak filter (see D1)
        if cand:
            ln, var, val = cand[rng.randrange(len(cand))]
            facts.append(Fact(
                "D3", {"intent": "D3", "var": var, "after_line": ln,
                       "occurrence": "first", "input": inp},
                val, "text"))

        # D4: does line N execute for input X (pick lines inside branches)
        branch_lines = _branch_stmt_lines(fn)
        if branch_lines:
            ln = branch_lines[rng.randrange(len(branch_lines))]
            facts.append(Fact(
                "D4", {"intent": "D4", "line": ln, "input": inp},
                "yes" if counts.get(ln, 0) > 0 else "no", "yesno"))

        # D5: which branch of the first if-with-else fires
        first_if = next((n for n in ast.walk(fn)
                         if isinstance(n, ast.If) and n.orelse), None)
        if first_if is not None and counts.get(first_if.lineno, 0) > 0:
            then_ln = first_if.body[0].lineno
            else_ln = first_if.orelse[0].lineno
            then_hit = counts.get(then_ln, 0) > 0
            else_hit = counts.get(else_ln, 0) > 0
            if then_hit != else_hit:
                facts.append(Fact(
                    "D5", {"intent": "D5", "if_line": first_if.lineno,
                           "input": inp},
                    "then" if then_hit else "else", "text"))

        # D6: loop iteration count for input X
        first_loop = next((n for n in ast.walk(fn)
                           if isinstance(n, (ast.For, ast.While))), None)
        if first_loop is not None:
            body_ln = first_loop.body[0].lineno
            facts.append(Fact(
                "D6", {"intent": "D6", "loop_line": first_loop.lineno,
                       "input": inp},
                str(counts.get(body_ln, 0)), "number"))

    # D7: exception prediction [HELD OUT]. Only from runs whose inputs pass
    # the function's own type expectations (typed_ok: every param kind
    # grounded in annotation/default/usage) — otherwise the "predicted"
    # exception is a wrong-typed-input artifact, not comprehension.
    typed_exc = [r for r in exc_runs if r.get("typed_ok")]
    if typed_exc:
        r = typed_exc[0]
        facts.append(Fact("D7", {"intent": "D7", "input": _fmt_input(r)},
                          r["exc"], "name", held_out=True))

    # C2-exec: situation-scoped behavior for the empty-input combo
    empty = next((r for r in runs if r.get("empty_param")
                  and r.get("status") in ("ok", "exception")), None)
    if empty is not None:
        ans = empty.get("result") if empty["status"] == "ok" \
            else f"raises {empty['exc']}"
        if ans:
            facts.append(Fact(
                "C2", {"intent": "C2", "param": empty["empty_param"],
                       "situation": "empty",
                       "other_args": _fmt_input(empty)},
                ans, "text", leak_by_design=True))
    return facts


def _branch_stmt_lines(fn: ast.AST) -> list[int]:
    lines = []
    for n in ast.walk(fn):
        if isinstance(n, ast.If):
            lines.append(n.body[0].lineno)
            if n.orelse:
                lines.append(n.orelse[0].lineno)
    return sorted(set(lines))


# exec intent metadata for the coverage report + triple building
EXEC_INTENTS = {
    "D1": "output of f(input)",
    "D2": "input producing output Y",
    "D3": "value of var V after line N",
    "D4": "does line N execute for input X",
    "D5": "which branch fires for input X",
    "D6": "loop iteration count for input X",
    "D7": "exception prediction [HELD OUT]",
}

if REGISTRY is not None:
    for _iid, _desc in EXEC_INTENTS.items():
        REGISTRY[_iid] = IntentSpec(
            _iid, "D", "exec", lambda ctx: False, lambda ctx: [],
            held_out=(_iid == "D7"), description=_desc,
        )


if __name__ == "__main__":
    if CHILD_FLAG in sys.argv:
        _child_main()
    else:
        print("qa_exec is a library + sandbox child; run qa_coverage.py",
              file=sys.stderr)
        sys.exit(1)
