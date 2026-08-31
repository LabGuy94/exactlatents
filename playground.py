#!/usr/bin/env python
"""Interactive playground for the released exact-latent model.

Load or paste a Python function, encode it into continuous vectors, and watch
the finetuned Qwen3-1.7B decoder reconstruct the exact source from the vectors
alone—or answer questions from them. Every result is compared against two
controls in the same session:

  vec    finetuned decoder reading continuous vectors (the system under test:
         encoder -> latent pooling @pf -> projector -> [<block>]vecs[</block>])
  text   the same finetuned decoder reading the raw code text
  stock  an untouched Qwen/Qwen3-1.7B reading the raw code text

At pf=4 the vector arm uses about four times fewer decoder context slots than
raw token text. This describes context slots, not storage size.

Run:  cd <repo root> && .venv/bin/python playground.py
Type 'demo' for a 60-second tour, 'help' for commands. See PLAYGROUND.md for a
worked example.

Everything printed by the REPL is also written without terminal colors to
playground_logs/session_<timestamp>.log.
"""

import argparse
import contextlib
import difflib
import gzip
import io
import itertools
import json
import math
import os
import re
import select
import shlex
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_ap = argparse.ArgumentParser(
    prog="playground.py",
    description="Interactive REPL for the released exact-latent model: "
                "reconstruct Python functions from continuous vectors using "
                "fewer decoder context slots, then compare with text controls.",
    epilog="REPL commands: demo | load canary <i|name> | load ood <i> | "
           "load qa <i> | paste | show | pf <n> | arms <list> | recon | "
           "compare | ask <question> | gold | raw | help | quit.  "
           "See PLAYGROUND.md.")
_ap.add_argument("--arms", default=None, metavar="LIST",
                 help="starting arms, comma-separated subset of vec,text,stock "
                      "(default: all three)")
_ap.add_argument("--pf", type=int, default=4, metavar="N",
                 help="starting pooling factor (default 4; 4 and 8 are the "
                      "trained factors)")
_ap.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                 help="compute device (default: auto-detect cuda > mps > cpu; "
                      "cpu runs in fp32)")
_ap.add_argument("--no-color", action="store_true",
                 help="disable ANSI colors even on a tty")
ARGS, _ = _ap.parse_known_args()

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_REAL_STDOUT = sys.stdout
USE_COLOR = (not ARGS.no_color) and (
    _REAL_STDOUT.isatty() or os.environ.get("PG_COLOR") == "1")


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def bold(s):    return _c("1", s)
def dim(s):     return _c("2", s)
def red(s):     return _c("31", s)
def boldred(s): return _c("1;31", s)
def green(s):   return _c("32", s)
def boldgrn(s): return _c("1;32", s)
def yellow(s):  return _c("1;33", s)


_ARM_C = {"vec": "1;36", "text": "1;34", "stock": "1;35"}


def badge(arm):
    """Consistent colored [vec]/[text]/[stock] chip."""
    return _c(_ARM_C.get(arm, "1"), f"[{arm}]")


class PgError(Exception):
    """A user-facing problem (missing file, bad state) — printed in plain
    English, never as a traceback."""


def _require(path, what, hint):
    if not Path(path).exists():
        raise PgError(f"{what} not found:\n    {path}\n  {hint}")


# --------------------------------------------------------------- transcript
class _Tee:
    """Mirror stdout to an ANSI-free session log.

    The bare ``pg> `` prompt is skipped because the REPL loop logs complete
    commands. Everything except ``write`` and ``flush`` delegates to the real
    stream so terminal input and line-editing integrations continue to work.
    """

    def __init__(self, real, log):
        self._real, self._log = real, log

    def write(self, s):
        st = _SPIN["cur"]
        if st is not None:
            with _SPIN_LOCK:
                if st["drawn"]:
                    # a spinner frame occupies the line — erase it first so
                    # normal output never shares a line with the spinner
                    self._real.write("\r\x1b[2K")
                    st["drawn"] = False
                self._real.write(s)
                # partial line pending (print(..., end=""))? hold the spinner
                # until the completing write arrives, or it lands mid-message
                st["hold"] = not s.endswith("\n")
        else:
            self._real.write(s)
        if s != "pg> ":
            self._log.write(ANSI_RE.sub("", s))
        return len(s)

    def flush(self):
        self._real.flush()
        self._log.flush()

    def __getattr__(self, name):  # fileno, isatty, encoding, errors, buffer, …
        return getattr(self._real, name)


_LOG = None  # session log file handle (set in main)


def _log_line(s):
    if _LOG:
        _LOG.write(s + "\n")
        _LOG.flush()


# ---------------------------------------------------------------- spinner
# ONE status line, updated strictly in place on the SAME terminal stream as
# all other output (the real stdout under the Tee — bypasses the transcript).
# Every tick: "\r\x1b[2K" + text, no newline. Any normal print while the
# spinner runs erases the frame first (see _Tee.write) and, when it leaves a
# partial line (end=""), holds the spinner until the line completes — the
# spinner and other output NEVER share a line. Cleanup is guaranteed in
# finally (exception/ctrl-C included). Non-tty: no spinner at all.
_SPIN = {"cur": None}
_SPIN_LOCK = threading.Lock()


@contextlib.contextmanager
def spinner(label):
    if not _REAL_STDOUT.isatty():
        yield
        return
    state = {"stop": False, "hold": False, "drawn": False}
    t0 = time.perf_counter()

    def run():
        frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        while True:
            time.sleep(0.09)
            with _SPIN_LOCK:
                if state["stop"]:
                    return
                if state["hold"]:
                    continue
                _REAL_STDOUT.write(
                    f"\r\x1b[2K\x1b[2m{next(frames)} {label} "
                    f"{time.perf_counter() - t0:4.0f}s\x1b[0m")
                _REAL_STDOUT.flush()
                state["drawn"] = True

    t = threading.Thread(target=run, daemon=True)
    _SPIN["cur"] = state
    t.start()
    try:
        yield
    finally:
        with _SPIN_LOCK:
            state["stop"] = True
            _SPIN["cur"] = None
            if state["drawn"]:
                _REAL_STDOUT.write("\r\x1b[2K")
                _REAL_STDOUT.flush()
        t.join(timeout=1)


# ------------------------------------------------------- syntax highlighting
_PYGMENTS = False
try:
    import pygments as _pyg
    from pygments.lexers import PythonLexer as _PygLex
    from pygments.formatters import Terminal256Formatter as _PygFmt
    _pyg_lex = _PygLex(stripnl=False, ensurenl=False)
    _pyg_fmt = _PygFmt(style="monokai")
    _PYGMENTS = True
except Exception:
    pass


def _hl_tokenize(text):
    """Stdlib fallback highlighter: keywords/strings/comments/numbers via
    tokenize. Robust to syntactically-broken generations — tokens collected
    up to the first error still get colored, the rest stays plain."""
    import keyword
    import tokenize as T
    lines = text.split("\n")
    spans = {}  # row -> [(c1, c2, color)]

    def add(r, c1, c2, col):
        if 1 <= r <= len(lines):
            spans.setdefault(r, []).append((c1, c2, col))

    try:
        for tok in T.generate_tokens(io.StringIO(text).readline):
            tt, ts, (r1, c1), (r2, c2), _ = tok
            col = ("32" if tt == T.STRING else "2" if tt == T.COMMENT
                   else "36" if tt == T.NUMBER
                   else "35" if tt == T.NAME and keyword.iskeyword(ts) else None)
            if not col:
                continue
            if r1 == r2:
                add(r1, c1, c2, col)
            else:
                add(r1, c1, len(lines[r1 - 1]), col)
                for r in range(r1 + 1, r2):
                    add(r, 0, len(lines[r - 1]), col)
                add(r2, 0, c2, col)
    except (T.TokenError, IndentationError, SyntaxError, ValueError):
        pass  # partial coloring is fine; uncolorable tail stays plain
    out = []
    for i, line in enumerate(lines, start=1):
        res, pos = [], 0
        for c1, c2, col in sorted(spans.get(i, [])):
            if c1 < pos:
                continue
            res.append(line[pos:c1])
            res.append(f"\x1b[{col}m{line[c1:c2]}\x1b[0m")
            pos = c2
        res.append(line[pos:])
        out.append("".join(res))
    return out


def hl_lines(text):
    """text -> list of syntax-highlighted lines (same count as input lines).
    Plain lines when colors are off or anything at all goes wrong."""
    plain = text.split("\n")
    if not USE_COLOR:
        return plain
    try:
        if _PYGMENTS:
            lines = _pyg.highlight(text, _pyg_lex, _pyg_fmt).split("\n")
            if len(lines) == len(plain) + 1 and not ANSI_RE.sub("", lines[-1]):
                lines.pop()
            return lines if len(lines) == len(plain) else plain
        return _hl_tokenize(text)
    except Exception:
        return plain


def vlen(s):
    return len(ANSI_RE.sub("", s))


def vpad(s, n):
    return s + " " * max(0, n - vlen(s))


def vtrunc(s, n):
    """Truncate to n visible chars, ANSI-aware, with a trailing ellipsis."""
    if vlen(s) <= n:
        return s
    out, vis, i = [], 0, 0
    while i < len(s) and vis < n - 1:
        m = ANSI_RE.match(s, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        out.append(s[i])
        vis += 1
        i += 1
    return "".join(out) + ("\x1b[0m" if USE_COLOR else "") + "…"


def clipw(s, reserve=0):
    """Clip to terminal width minus reserve with a dim ellipsis — gutter-
    aligned code lines must NEVER wrap (wrapping shatters the layout; 'raw'
    side-by-side is where full content lives, and even that clips per panel)."""
    w = shutil.get_terminal_size().columns
    return vtrunc(s, max(20, w - reserve))


def _pair_hl(x, y):
    """Git-style char-level pair highlight: returns (old_line, new_line) with
    the old rendered red / new rendered green and the exact differing char
    spans marked with a background color."""
    sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
    xs, ys = [], []
    for t, i1, i2, j1, j2 in sm.get_opcodes():
        if t == "equal":
            xs.append(red(x[i1:i2]))
            ys.append(green(y[j1:j2]))
        else:
            if i2 > i1:
                xs.append(_c("41;97", x[i1:i2]))  # bg-red on removed chars
            if j2 > j1:
                ys.append(_c("42;30", y[j1:j2]))  # bg-green on added chars
    return "".join(xs), "".join(ys)


print(dim("· starting up (importing torch/transformers) …"), flush=True)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
    _hf_logging.disable_progress_bar()
except Exception:
    pass

from compressor import ft_load  # noqa: E402
from compressor.exactness import byte_exact, code_exact  # noqa: E402
from compressor.genutil import batched_generate  # noqa: E402

BASE = "Qwen/Qwen3-1.7B"
MODEL_STATE = ROOT / "weights/model.safetensors"
PROJECTOR_EMA = ROOT / "weights/projector_ema.safetensors"
CANARIES = ROOT / "receipts/canaries/pf4_gens.jsonl"
OOD = ROOT / "receipts/ood600/ood600_pf4_gens.jsonl"
QA = ROOT / "receipts/ood_qa/questions_frozen.jsonl.gz"
LOG_DIR = ROOT / "playground_logs"

WEIGHTS_HINT = (
    "Download the released weights with:\n"
    "    hf download labguy/exactlatents-qwen3-1.7b --local-dir weights/"
)
DATA_HINT = "This file ships with the repository; check your checkout."

def _pick_device():
    """(device, dtype, banner_line). Auto-detect, --device overrides."""
    want = ARGS.device
    if want == "cuda" and not torch.cuda.is_available():
        print(yellow("--device cuda requested but CUDA is unavailable — auto-detecting"))
        want = None
    if want == "mps" and not torch.backends.mps.is_available():
        print(yellow("--device mps requested but MPS is unavailable — auto-detecting"))
        want = None
    dev = want or ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "cuda":
        return dev, torch.bfloat16, f"cuda ({torch.cuda.get_device_name(0)}) · bf16"
    if dev == "mps":
        return dev, torch.bfloat16, "mps (Apple GPU via PyTorch) · bf16"
    forced = " (forced via --device)" if want == "cpu" else \
        " (no GPU found — generation will be slow)"
    return dev, torch.float32, f"cpu · fp32{forced}"


DEVICE, DTYPE, DEVICE_DESC = _pick_device()
ALL_ARMS = ["vec", "text", "stock"]

S = {"pf": 4, "arms": list(ALL_ARMS), "rec": None,
     "last_raw": {}, "last_kind": None, "stack": None}
_m = {"load_log": []}  # lazily loaded heavyweights + captured provenance lines


# ---------------------------------------------------------------- lazy loading
@contextlib.contextmanager
def _load_step(msg):
    """One dim line per load step: '· loading X … done (6s)'. Library prints
    ([ft_load] provenance, HF chatter) are captured into _m['load_log']."""
    print(dim(f"· {msg} …"), end="", flush=True)
    t0 = time.perf_counter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield
    _m["load_log"] += buf.getvalue().splitlines()
    print(dim(f" done ({time.perf_counter() - t0:.0f}s)"), flush=True)


def _free_ft_cache():
    """Drop the cached state once the decoder and encoder have been loaded."""
    if "ft_dec" in _m and "comp" in _m:
        ft_load._CACHE.clear()
        import gc
        gc.collect()


def get_tok():
    if "tok" not in _m:
        _m["tok"] = AutoTokenizer.from_pretrained(BASE)
    return _m["tok"]


def get_ft_dec():
    if "ft_dec" not in _m:
        _require(MODEL_STATE, "finetuned weights (weights/model.safetensors)",
                 WEIGHTS_HINT)
        with _load_step("loading finetuned decoder"):
            dec = AutoModelForCausalLM.from_pretrained(
                BASE, dtype=DTYPE).to(DEVICE).eval()
            ft_load.load_decoder(dec, str(MODEL_STATE))
        _m["ft_dec"] = dec
        _free_ft_cache()
    return _m["ft_dec"]


def get_stock_dec():
    if "stock_dec" not in _m:
        with _load_step("loading stock decoder"):
            _m["stock_dec"] = AutoModelForCausalLM.from_pretrained(
                BASE, dtype=DTYPE).to(DEVICE).eval()
    return _m["stock_dec"]


def get_comp():
    """Build the released compressor at the requested pooling factor."""
    if "comp" not in _m:
        _require(MODEL_STATE, "finetuned weights (weights/model.safetensors)",
                 WEIGHTS_HINT)
        _require(PROJECTOR_EMA, "projector EMA (weights/projector_ema.safetensors)",
                 WEIGHTS_HINT)
        from compressor import Compressor
        with _load_step("loading compressor (encoder + latent pooler + projector)"):
            dec_hidden = AutoConfig.from_pretrained(BASE).hidden_size
            comp = Compressor(encoder_name=BASE, decoder_hidden=dec_hidden,
                              pooling_factor=S["pf"], proj_width=None, proj_depth=2,
                              pooling="latent", boundary=True,
                              bidirectional=False, dtype=DTYPE).to(DEVICE)
            ft_load.load_projector(comp.projector, PROJECTOR_EMA, prefer="ema")
            ft_load.load_encoder(comp, str(MODEL_STATE))
            comp.eval()
        _m["comp"] = comp
        _free_ft_cache()
    return _m["comp"]


def get_canaries():
    if "canaries" not in _m:
        _require(CANARIES, "released canary records", DATA_HINT)
        _m["canaries"] = [json.loads(line) for line in open(CANARIES)]
    return _m["canaries"]


def get_ood():
    if "ood" not in _m:
        _require(OOD, "released OOD-600 records", DATA_HINT)
        _m["ood"] = [json.loads(line) for line in open(OOD)]
    return _m["ood"]


def get_qa():
    """Load the released fully OOD QA records."""
    if "qa" not in _m:
        _require(QA, "released OOD QA questions", DATA_HINT)
        with gzip.open(QA, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        for row in rows:
            row["answer"] = row["gold"]
        _m["qa"] = rows
        print(dim(f"  {len(rows)} fully OOD QA records ready"))
    return _m["qa"]




# ---------------------------------------------------------------- record state
def _size_bucket(n):
    return ("tiny" if n < 32 else "small" if n < 64 else "mid" if n < 256
            else "large" if n < 1024 else "XL")


def set_rec(name, code, src, qa=None, membership=None):
    if S.get("stack"):
        S["stack"] = None
        print(dim("(left stack mode — single function loaded; rebuild with "
                  "'stack add')"))
    tok = get_tok()
    n = len(tok(code, add_special_tokens=False).input_ids)
    S["rec"] = {"name": name, "code": code, "src": src, "qa": qa, "n_tok": n}
    S["last_raw"], S["last_kind"] = {}, None
    print(f"loaded [{src}] {bold(name)}  ({n} tokens, {len(code)} bytes"
          + (f", gold QA intent {qa['intent']}" if qa else "") + ")")
    g = math.ceil(n / S["pf"])
    print(dim(f"  ≈{g} vectors (+2 block markers) @pf{S['pf']} · "
              f"size {_size_bucket(n)} ({n} tok)"
              f" · {membership or 'membership unknown'}"))
    if qa:
        print(f"  gold Q: {qa['question']}")
        print(f"  gold A: {qa['answer']}")
    print(dim("  try: recon · ask <question>"))


def need_rec():
    if S["rec"] is None:
        print("no function loaded — try: demo · load canary 5 · paste")
        return None
    return S["rec"]


def _browse_rows(kind):
    """Build picker rows for one released record set."""
    rows = []
    if kind == "canary":
        for i, c in enumerate(get_canaries()):
            rows.append({"i": i, "label": c["func_name"],
                         "meta": f"{len(c['code']):>5} bytes",
                         "search": c["func_name"], "code": c["code"]})
    elif kind == "ood":
        for i, c in enumerate(get_ood()):
            rows.append({"i": i, "label": c["func_name"],
                         "meta": f"{len(c['code']):>5} bytes",
                         "search": c["func_name"], "code": c["code"]})
    else:  # qa
        for i, r in enumerate(get_qa()):
            q = " ".join(str(r["question"]).split())
            intent = r.get("intent", "QA")
            rows.append({"i": i, "label": f"{intent} · {q[:58]}",
                         "meta": r.get("answer_type", ""),
                         "search": f"{intent} {q}", "code": r["code"]})
    return rows


def _row_code(row):
    return row["code"]


def cmd_browse(kind):
    """Full-screen picker: arrows/j-k move, PgUp/PgDn or ctrl-u/d page,
    printable chars filter by name substring, Enter selects, ESC clears the
    filter (or cancels), q cancels. Returns the selected index or None.
    Falls back to a plain numbered list when stdin isn't a tty."""
    rows = _browse_rows(kind)
    if not (sys.stdin.isatty() and _REAL_STDOUT.isatty()):
        show = rows[:40]
        for r in show:
            print(f"  {r['i']:>4}  {r['label'][:56]:<56}  {r['meta']}")
        if len(rows) > len(show):
            print(dim(f"  … {len(rows)} total — showing the first {len(show)}; "
                      f"'load {kind} <idx>' reaches the rest"))
        try:
            s = input("index (blank to cancel): ").strip()
        except EOFError:
            return None
        return int(s) if s.isdigit() and 0 <= int(s) < len(rows) else None

    import select as _select
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    out = _REAL_STDOUT
    filt, cur, top = "", 0, 0
    try:
        tty.setcbreak(fd)
        out.write("\x1b[?1049h\x1b[?25l")  # alt screen, hide cursor
        while True:
            f = filt.lower()
            vis = [r for r in rows if f in r["search"].lower()] if f else rows
            cur = max(0, min(cur, len(vis) - 1)) if vis else 0
            size = shutil.get_terminal_size()
            w, h = size.columns, size.lines
            prev_h = 12 if h >= 26 else max(0, h - 12)
            list_h = max(3, h - prev_h - 3)
            if cur < top:
                top = cur
            if cur >= top + list_h:
                top = cur - list_h + 1
            out.write("\x1b[H")
            title = (f" pick: {kind} — ↑/↓ (or j/k) move · PgUp/PgDn page · "
                     "type to filter · Enter select · ESC/q cancel")
            out.write("\x1b[1m" + title[: w - 1] + "\x1b[0m\x1b[K\r\n")
            out.write(f"\x1b[2m filter: {filt}▏  {len(vis)}/{len(rows)} "
                      "shown\x1b[0m\x1b[K\r\n")
            for k in range(list_h):
                idx = top + k
                if idx < len(vis):
                    r = vis[idx]
                    line = f" {r['i']:>4}  {r['label'][:58]:<58}  {r['meta']}"[: w - 1]
                    if idx == cur:
                        out.write("\x1b[7m" + line + "\x1b[27m")
                    else:
                        out.write(line)
                out.write("\x1b[K\r\n")
            if prev_h > 0 and vis:
                out.write("\x1b[2m ── preview ──────────────────\x1b[0m\x1b[K\r\n")
                for pn, pl in enumerate(hl_lines(_row_code(vis[cur]))[: prev_h - 1],
                                        start=1):
                    out.write(" \x1b[2m" + f"{pn:>3}│\x1b[0m "
                              + vtrunc(pl, w - 8) + "\x1b[K\r\n")
            out.write("\x1b[J")
            out.flush()
            ch = os.read(fd, 1)
            if os.environ.get("PG_DEBUG_KEYS"):
                with open(os.environ["PG_DEBUG_KEYS"], "ab") as kf:
                    kf.write(ch)
            if ch == b"\x1b":
                r_, _, _ = _select.select([fd], [], [], 0.05)
                if r_:
                    seq = os.read(fd, 2)
                    if seq == b"[A":
                        cur -= 1
                    elif seq == b"[B":
                        cur += 1
                    elif seq == b"[5":
                        os.read(fd, 1)
                        cur -= list_h
                    elif seq == b"[6":
                        os.read(fd, 1)
                        cur += list_h
                else:  # bare ESC
                    if filt:
                        filt, cur = "", 0
                    else:
                        return None
            elif ch in (b"\r", b"\n"):
                return vis[cur]["i"] if vis else None
            elif ch in (b"\x7f", b"\x08"):
                filt = filt[:-1]
            elif ch == b"\x15":  # ctrl-u
                cur -= list_h
            elif ch == b"\x04":  # ctrl-d
                cur += list_h
            elif ch == b"q" and not filt:
                return None
            elif ch == b"k" and not filt:
                cur -= 1
            elif ch == b"j" and not filt:
                cur += 1
            elif ch == b"\x03":
                return None  # belt — normally raises KeyboardInterrupt
            elif ch and 32 <= ch[0] < 127:
                filt += ch.decode()
                cur = 0
    except KeyboardInterrupt:
        return None
    finally:  # restore on EVERY exit path, ^C included
        out.write("\x1b[?25h\x1b[?1049l")
        out.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def cmd_load(args):
    if len(args) == 1 and args[0] in ("canary", "ood", "qa"):
        sel = cmd_browse(args[0])
        if sel is None:
            print("(cancelled)")
            return
        _log_line(f"(picker selected {args[0]} {sel})")
        args = [args[0], str(sel)]
    if len(args) < 2:
        print("usage: load canary <idx|name> | load ood <idx> | load qa <idx>  "
              "(no index = interactive picker)")
        return
    kind, key = args[0], " ".join(args[1:])
    if kind == "canary":
        cans = get_canaries()
        if key.isdigit():
            i = int(key)
            if not 0 <= i < len(cans):
                print(f"canary index 0..{len(cans) - 1}")
                return
            c = cans[i]
        else:
            hits = [c for c in cans if key in c["func_name"]]
            if not hits:
                print(f"no canary matching '{key}'. names:")
                print("  " + ", ".join(c["func_name"] for c in cans))
                return
            c = hits[0]
        set_rec(c["func_name"], c["code"], f"canary {cans.index(c)}",
                membership=f"canary set, tier '{c.get('tier', 'v1')}' "
                           "(held-out repositories)")
    elif kind == "ood":
        ood = get_ood()
        if not key.isdigit() or not 0 <= int(key) < len(ood):
            print(f"usage: load ood <0..{len(get_ood()) - 1}>")
            return
        c = ood[int(key)]
        set_rec(c["func_name"], c["code"], f"ood {int(key)}",
                membership="OOD-600 post-cutoff set")
    elif kind == "qa":
        qa = get_qa()
        if not key.isdigit() or not 0 <= int(key) < len(qa):
            print(f"usage: load qa <0..{len(qa) - 1}>")
            return
        r = qa[int(key)]
        code = r["code"]
        name = r.get("func_name") or next(
            (ln.split("def ", 1)[1].split("(")[0].strip()
             for ln in code.split("\n") if "def " in ln),
            f"qa_{int(key)}")
        set_rec(name, code, f"qa {int(key)}", qa=r,
                membership="fully OOD QA questions")
    else:
        print("usage: load canary <idx|name> | load ood <idx> | load qa <idx>")


# ------------------------------------------------------------- paste capture
# Pasted lines are read directly from stdin rather than through the line
# editor. This prevents a redraw per line during fast multi-line pastes and
# keeps the transcript byte-for-byte faithful to the captured input.
_STDIN_BUF = bytearray()


def _stdin_line_tty(timeout, flush_partial=False):
    """One line from the tty fd via os.read + own buffering (no read-ahead
    stealing from libedit). None on quiet-timeout/EOF; flush_partial returns
    a trailing unterminated fragment (e.g. 'lastline<ESC>[201~') on quiet."""
    fd = sys.stdin.fileno()
    while True:
        nl = _STDIN_BUF.find(b"\n")
        if nl >= 0:
            ln = _STDIN_BUF[:nl].decode("utf-8", "replace")
            del _STDIN_BUF[: nl + 1]
            return ln
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            if flush_partial and _STDIN_BUF:
                ln = _STDIN_BUF.decode("utf-8", "replace")
                _STDIN_BUF.clear()
                return ln
            return None
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            return None
        if not chunk:  # EOF
            if _STDIN_BUF:
                ln = _STDIN_BUF.decode("utf-8", "replace")
                _STDIN_BUF.clear()
                return ln
            return None
        _STDIN_BUF.extend(chunk)


def _stdin_pending(timeout=0):
    if not sys.stdin.isatty():
        return False
    if _STDIN_BUF:
        return True
    r, _, _ = select.select([sys.stdin.fileno()], [], [], timeout)
    return bool(r)


def _strip_paste_markers(ln):
    """Remove bracketed-paste markers (\\x1b[200~ / \\x1b[201~), including
    libedit-mangled residues. Returns (clean_line, saw_end_marker)."""
    end = False
    for pre in ("\x1b[200~", "[200~", "200~", "00~", "0~"):
        if ln.startswith(pre):
            ln = ln[len(pre):]
            break
    for suf in ("\x1b[201~", "[201~", "201~"):
        if ln.endswith(suf):
            ln = ln[: -len(suf)]
            end = True
            break
    return ln, end


def _capture_paste(first_line=None, explicit=True):
    """Collect pasted lines until the lone-'.' terminator, a bracketed-paste
    end marker, or (auto-detected pastes on a tty) a 0.35s quiet gap after
    the burst. explicit=True (the `paste` command) always waits for '.'."""
    lines = []
    if first_line is not None:
        ln, end = _strip_paste_markers(first_line)
        _log_line(ln)
        if ln.strip() != ".":
            lines.append(ln)
        if end:
            return "\n".join(lines)
    tty = sys.stdin.isatty()
    while True:
        if tty:
            ln = _stdin_line_tty(None if explicit else 0.35,
                                 flush_partial=not explicit)
            if ln is None:
                break  # quiet gap (auto mode) or EOF
        else:
            raw = sys.stdin.readline()
            if not raw:
                break
            ln = raw.rstrip("\n")
        ln, end = _strip_paste_markers(ln)
        _log_line(ln)
        if ln.strip() == ".":
            break
        lines.append(ln)
        if end:
            break
    return "\n".join(lines)


def _dedent_if_needed(code):
    """Class methods arrive indented; ast-based grading (code_exact)
    fail-closes on IndentationError, so a 99.9%-similar reconstruction would
    read as MISS. Dedent BEFORE storing so compression, generation, grading,
    and display all see the same top-level form."""
    import ast
    import textwrap
    try:
        ast.parse(code)
        return code, False
    except (SyntaxError, IndentationError):
        ded = textwrap.dedent(code)
        if ded != code:
            try:
                ast.parse(ded)
                return ded, True
            except (SyntaxError, IndentationError):
                pass
    return code, False


def _finalize_paste(code, auto=False):
    code = code.rstrip()
    if not code.strip():
        print("(nothing pasted)")
        return
    n_lines = code.count("\n") + 1
    print(f"(captured {n_lines} lines, {len(code)} bytes — "
          "'show' displays the clean capture)")
    if auto:
        # pasting AT the prompt routes the first line through libedit in raw
        # mode; on very large machine-speed pastes the kernel's raw input
        # queue can overflow and silently drop bytes (pty-measured: clean at
        # ~3KB, one torn region at 10KB). The `paste` command reads in
        # canonical mode with real backpressure and is byte-exact at any size.
        import ast
        try:
            ast.parse(code)
        except (SyntaxError, IndentationError):
            try:
                ast.parse(__import__("textwrap").dedent(code))
            except (SyntaxError, IndentationError):
                print(yellow("  ⚠ the captured code does not parse — very large "
                             "pastes at the prompt can drop bytes.\n    For big "
                             "functions type `paste` first; that path is "
                             "byte-exact at any size."))
    code, dedented = _dedent_if_needed(code)
    if dedented:
        print(dim("(dedented pasted code — it was indented as a class method)"))
    name = next((ln.split("def ", 1)[1].split("(")[0].strip()
                 for ln in code.split("\n") if "def " in ln), "pasted")
    set_rec(name, code, "paste", membership="custom (not in dataset)")


def cmd_paste():
    print("paste your function, then a line with just a single .  to finish:")
    sys.stdout.flush()
    _finalize_paste(_capture_paste(explicit=True))


_KNOWN_CMDS = {"demo", "load", "paste", "show", "pf", "arms", "recon", "stack",
               "compare", "sample", "ask", "gold", "raw", "help", "quit",
               "exit", "q"}


def _paste_event(raw_line):
    """Did the prompt read the FIRST line of a paste instead of a command?
    Bracketed-paste markers are definitive; on a tty, code-looking first
    lines and immediately-buffered follow-up input are the fallback (libedit
    sometimes mangles the markers)."""
    if raw_line.lstrip().startswith(("\x1b[200~", "[200~", "200~")):
        return True
    if not sys.stdin.isatty():
        return False
    ls = raw_line.strip()
    if ls.startswith(("def ", "async def ", "@")) or ls.endswith(":"):
        return True
    first = (ls.split() or [""])[0].lower()
    return first not in _KNOWN_CMDS and _stdin_pending(0)


# ---------------------------------------------------------------- stack mode
# Each function is preceded by ``### function: {name}\n`` and then its vector
# block (or raw code text for the text controls). Generation is prompted with
# ``\nReproduce the function `{name}`:\n``. Short names keep headers compact.


def _short_name(name):
    return str(name).split(".")[-1]




def _fetch_fn(kind, key):
    """(name, code) for a canary/ood/qa record, for stack building."""
    if kind == "canary":
        cans = get_canaries()
        if key.isdigit() and 0 <= int(key) < len(cans):
            c = cans[int(key)]
        else:
            hits = [c for c in cans if key and key in c["func_name"]]
            if not hits:
                raise PgError(f"no canary matching '{key}' (0-{len(cans) - 1} or name part)")
            c = hits[0]
        return c["func_name"], c["code"]
    if kind == "ood":
        ood = get_ood()
        if not (key.isdigit() and 0 <= int(key) < len(ood)):
            raise PgError(f"ood index must be 0-{len(ood) - 1}")
        c = ood[int(key)]
        return c["func_name"], c["code"]
    if kind == "qa":
        qa = get_qa()
        if not (key.isdigit() and 0 <= int(key) < len(qa)):
            raise PgError(f"qa index must be 0-{len(qa) - 1}")
        r = qa[int(key)]
        return r["func_name"], r["code"]
    raise PgError("stack add takes: canary <i|name> | ood <i> | qa <i> | paste")


def _mk_member(name, code):
    n = len(get_tok()(code, add_special_tokens=False).input_ids)
    return {"name": _short_name(name), "code": code, "n_tok": n}


def _stack_est(members):
    """(total_tokens, total_vectors_at_pf) across members."""
    tot = sum(m["n_tok"] for m in members)
    vec = sum(math.ceil(m["n_tok"] / S["pf"]) + 2 for m in members)
    return tot, vec


def _stack_warn(members):
    tot, vec = _stack_est(members)
    if tot > 6000:
        print(yellow(f"  ⚠ large stack: {tot} source tokens — text arms exceed "
                     "the 6144-token training budget and generation will be slow"))
    elif vec + 8 * len(members) > 4000:
        print(yellow(f"  ⚠ large stack: ~{vec} vectors — generation will be slow"))


def _stack_summary():
    st = S["stack"]
    tot, vec = _stack_est(st["members"])
    ti = st.get("target_idx")
    tgt = f" · target #{ti} {st['members'][ti]['name']}" if ti is not None else ""
    print(f"stack: {len(st['members'])} fns · {tot} tok → {vec} vec @pf{S['pf']}"
          f"{tgt} · {st['src']}")
    _stack_warn(st["members"])


def cmd_stack(args):
    if not args:
        if S["stack"]:
            _stack_summary()
        else:
            print("usage: stack add canary <i>|ood <i>|qa <i>|paste · "
                  "stack rm <name|idx> · stack clear · stack show")
        return
    sub = args[0]
    if sub == "add":
        if len(args) < 2:
            print("usage: stack add canary <i|name> | ood <i> | qa <i> | paste")
            return
        if args[1] == "paste":
            print("paste the function, then a line with just a single .  to finish:")
            sys.stdout.flush()
            code = _capture_paste(explicit=True).rstrip()
            if not code.strip():
                print("(nothing pasted)")
                return
            code, dedented = _dedent_if_needed(code)
            if dedented:
                print(dim("(dedented pasted code — it was indented as a class method)"))
            name = next((ln.split("def ", 1)[1].split("(")[0].strip()
                         for ln in code.split("\n") if "def " in ln), "pasted")
        else:
            name, code = _fetch_fn(args[1], " ".join(args[2:]))
        if S["stack"] is None:
            S["stack"] = {"members": [], "target_idx": None, "src": "manual"}
        S["stack"]["members"].append(_mk_member(name, code))
        _stack_summary()
    elif sub == "rm":
        st = S["stack"]
        if not st or not args[1:]:
            print("usage: stack rm <name|idx>   (stack show lists them)")
            return
        key = " ".join(args[1:])
        idx = (int(key) if key.isdigit() and 0 <= int(key) < len(st["members"])
               else next((i for i, m in enumerate(st["members"]) if m["name"] == key), None))
        if idx is None:
            print(f"no stack member '{key}'")
            return
        gone = st["members"].pop(idx)
        if st["target_idx"] is not None:
            if st["target_idx"] == idx:
                st["target_idx"] = None
            elif st["target_idx"] > idx:
                st["target_idx"] -= 1
        print(f"removed #{idx} {gone['name']}")
        if not st["members"]:
            S["stack"] = None
            print("(stack empty — left stack mode)")
        else:
            _stack_summary()
    elif sub == "clear":
        S["stack"] = None
        print("(stack cleared — back to single-function mode)")
    elif sub == "show":
        st = S["stack"]
        if not st:
            print("no stack — build one with: stack add canary <i>")
            return
        run = 0
        for i, m in enumerate(st["members"]):
            v = math.ceil(m["n_tok"] / S["pf"]) + 2
            run += v
            mark = " ◀ target" if st.get("target_idx") == i else ""
            print(f"  {i:>2}  {m['name']:<32.32} {m['n_tok']:>5} tok → {v:>4} vec "
                  f"(running {run}){mark}")
        _stack_summary()
    else:
        print(f"unknown stack subcommand '{sub}' — try: stack")



def _stack_ctx(arm, members, prompt_text):
    """Assemble one decoder context from the stack members."""
    dec = get_ft_dec() if arm in ("vec", "text") else get_stock_dec()
    tok = get_tok()
    emb_t = dec.get_input_embeddings()

    def embeds(text):
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(DEVICE)
        return emb_t(ids)

    parts = []
    with torch.no_grad():
        if arm == "vec":
            comp = get_comp()
            comp.pooling_factor = S["pf"]
            for m in members:
                parts.append(embeds(f"### function: {m['name']}\n"))
                parts.append(comp.compress(m["code"], DEVICE).to(emb_t.weight.dtype))
        else:
            for m in members:
                parts.append(embeds(f"### function: {m['name']}\n"))
                parts.append(embeds(m["code"]))
        parts.append(embeds(prompt_text))
        ctx = torch.cat(parts, dim=1)
    return dec, ctx[0]


def stack_recon(sel):
    st = S["stack"]
    members = st["members"]
    if sel is None:
        ti = st.get("target_idx")
        if ti is None:
            print("stack loaded — name the target: recon <name|idx>   (stack show lists them)")
            return
    elif sel.isdigit() and 0 <= int(sel) < len(members):
        ti = int(sel)
    else:
        hits = [i for i, m in enumerate(members) if m["name"] == sel]
        if not hits:
            hits = [i for i, m in enumerate(members) if sel in m["name"]]
        if not hits:
            print(f"no stack member matching '{sel}' — stack show lists them")
            return
        if len(hits) > 1:
            print(yellow(f"  {len(hits)} members named '{sel}' "
                         f"(#{', #'.join(map(str, hits))}) — using #{hits[0]}; "
                         "pick others by index"))
        ti = hits[0]
    tok = get_tok()
    tgt = members[ti]
    code = tgt["code"]
    true_ids = tok(code, add_special_tokens=False).input_ids
    budget = int(len(true_ids) * 1.3) + 16
    prompt = f"\nReproduce the function `{tgt['name']}`:\n"
    tot, vec = _stack_est(members)
    print(f"stack recon: target #{ti} {bold(tgt['name'])} ({tgt['n_tok']} tok) "
          f"from a {len(members)}-fn stack ({tot} tok → {vec} vec @pf{S['pf']}) · "
          f"arms: {' '.join(S['arms'])}")
    _stack_warn(members)
    for arm in S["arms"]:
        t0 = time.perf_counter()
        try:
            with spinner(f"{arm} stack recon"):
                dec, ctx = _stack_ctx(arm, members, prompt)
                out = batched_generate(dec, [ctx], [budget], batch_size=1)[0]
        except KeyboardInterrupt:
            print(f"\n{badge(arm)} interrupted — back to prompt")
            return
        dt = time.perf_counter() - t0
        gen = tok.decode(out.tolist(), skip_special_tokens=True)
        S["last_raw"][arm] = gen
        S["last_kind"] = "recon"
        S["rec"] = {"name": tgt["name"], "code": code, "src": f"stack #{ti}",
                    "qa": None, "n_tok": tgt["n_tok"]}
        be, ce, sim, gradeable = _grade(code, gen)
        tag, _ = _verdict_tags(be, ce, sim, gradeable)
        print(f"{vpad(badge(arm), 8)} {tag}   ctx {ctx.shape[0]}   {dt:.1f}s")
        if not be:
            # the collision failure mode: did it decode a DIFFERENT member?
            sib = next(((j, m) for j, m in enumerate(members) if j != ti
                        and (byte_exact(m["code"], gen) or code_exact(m["code"], gen))),
                       None)
            if sib:
                j, m = sib
                print(boldred(f"    ⚠ decoded WRONG SIBLING: {m['name']} "
                              f"(member #{j}) — the collision failure mode"))
                continue
            sims = sorted(((j, difflib.SequenceMatcher(
                None, m["code"], gen[: len(m["code"]) + 200], autojunk=False).ratio())
                for j, m in enumerate(members) if j != ti),
                key=lambda x: -x[1])[:2]
            if sims and sims[0][1] > sim + 0.10 and sims[0][1] > 0.55:
                j, s = sims[0]
                print(yellow(f"    ⚠ generation resembles member #{j} "
                             f"{members[j]['name']} (sim {100 * s:.0f}%) more than "
                             f"the target (sim {100 * sim:.0f}%) — likely sibling grab"))
            if sims:
                print(dim("    sibling check: closest members — "
                          + " · ".join(f"#{j} {members[j]['name']} sim {100 * s:.0f}%"
                                       for j, s in sims)
                          + f"  (target sim {100 * sim:.0f}%)"))
            print_diff(code, gen, ce)
    print(dim("('raw' compares the generations against the TARGET function)"))


def stack_ask(question):
    st = S["stack"]
    if not question:
        print("stack loaded — ask needs a question: ask <question>")
        return
    tok = get_tok()
    members = st["members"]
    prompt = f"\nQ: {question}\nA:"
    print(f"Q: {question}")
    for arm in S["arms"]:
        t0 = time.perf_counter()
        try:
            with spinner(f"{arm} stack answering"):
                dec, ctx = _stack_ctx(arm, members, prompt)
                out = batched_generate(dec, [ctx], [200], batch_size=1)[0]
        except KeyboardInterrupt:
            print(f"\n{badge(arm)} interrupted — back to prompt")
            return
        dt = time.perf_counter() - t0
        gen = tok.decode(out.tolist(), skip_special_tokens=True)
        S["last_raw"][arm] = gen
        S["last_kind"] = "ask"
        print(f"--- stack-QA {badge(arm)} ({dt:.1f}s, ctx {ctx.shape[0]}) "
              + dim("(exploratory)") + " ---")
        print_text_block(gen)


# ------------------------------------------------------------------ arms/gen
def arm_context(arm, code, name, prompt_ids=None, recon=False):
    """Build the embedding context for one arm. Returns (decoder, ctx, n_ctx)."""
    tok = get_tok()
    if arm == "vec":
        dec = get_ft_dec()
        comp = get_comp()
        comp.pooling_factor = S["pf"]
        with torch.no_grad():
            vecs = comp.compress(code, DEVICE)[0]  # [<block>]vecs[</block>]
        ctx = vecs
        if prompt_ids is not None:
            pemb = dec.get_input_embeddings()(
                torch.tensor(prompt_ids, device=DEVICE)).to(vecs.dtype)
            ctx = torch.cat([vecs, pemb], dim=0)
        return dec, ctx, vecs.shape[0]
    dec = get_ft_dec() if arm == "text" else get_stock_dec()
    if recon:
        pre = f"### function: {name}\n{code}\n### function: {name}\n"
        ids = tok(pre, add_special_tokens=False).input_ids
    else:
        ids = tok(code, add_special_tokens=False).input_ids + list(prompt_ids or [])
    ctx = dec.get_input_embeddings()(torch.tensor(ids, device=DEVICE))
    return dec, ctx, len(ids)


def print_diff(code, gen, ce):
    """Show ONLY genuinely differing lines, 1 line of context, colored, with
    the ORIGINAL's line numbers in the gutter.

    Display-only — verdicts (byte/code-exact) are computed on the raw
    generation before this runs and are never affected by this filtering.
    Diff lines stay plain red/green (no syntax colors — they would fight).

    Two noise sources are filtered:
    * OVERFLOW: the generation keeps running past the end of the function
      ('### function: …' babble). Everything past the last opcode that
      consumes original lines is dropped — including surplus added lines
      bundled INTO that final opcode, and continuation glued onto the last
      line when the source has no trailing newline — replaced by a dim note.
    * WHITESPACE: a -/+ pair whose lines are token-wise identical
      (x.split() == y.split(): trailing spaces, tab-vs-space, internal
      indent) would render as visually identical twins — collapsed to one
      dim count instead, pair by pair, even inside mixed diff blocks.
    """
    a = code.splitlines()
    b = gen[: len(code) + 200].splitlines()
    nw = max(2, len(str(len(a))))
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    # last opcode that consumes original lines: anything after it is overflow
    last_consume = max((k for k, (t, i1, i2, j1, j2) in enumerate(ops) if i2 > i1),
                       default=-1)
    ws_only = 0
    overflow = False
    blocks = []  # (ctx_lineno_or_None, [('-'|'+', a_lineno_or_None, line), ...])
    for k, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag == "equal":
            continue
        if k > last_consume:
            overflow = True  # insert(s) past the last original line
            continue
        rem, add = a[i1:i2], b[j1:j2]
        if k == last_consume and len(add) > len(rem):
            add = add[: len(rem)]  # surplus adds at the very end = overflow
            overflow = True
        pairs = []
        np_ = min(len(rem), len(add))
        for p, (x, y) in enumerate(zip(rem[:np_], add[:np_])):
            if x.split() == y.split():
                ws_only += 1  # visually identical — never render as -/+
            elif (k == last_consume and i2 == len(a) and p == np_ - 1
                  and x.strip() and (y.startswith(x) or y.startswith(x.rstrip()))):
                # last original line reproduced with continuation glued onto it
                # (source had no trailing newline) — overflow, not a mismatch
                overflow = True
            else:
                pairs += [("-", i1 + p + 1, x), ("+", None, y)]
        pairs += [("-", i1 + np_ + q + 1, x) for q, x in enumerate(rem[np_:])]
        pairs += [("+", None, y) for y in add[np_:]]
        if pairs:
            blocks.append((i1 if i1 > 0 else None, pairs))
    if not blocks and not ws_only and not overflow:
        return
    if ce and blocks:
        print(dim("    code identical — differences are prose/comments only:"))
    shown = 0
    total = sum(len(p) for _, p in blocks)
    for ctx_no, pairs in blocks:
        if shown >= 16:
            print(dim(f"    … ({total - shown} more differing lines — "
                      "'raw' shows the full output)"))
            break
        if ctx_no is not None:
            print(dim(f"      {ctx_no:>{nw}}│ " + clipw(a[ctx_no - 1], nw + 10)))
        for kind, no, l in pairs:
            if shown >= 16:
                break
            num = f"{no:>{nw}}" if no else " " * nw
            l = clipw(l, nw + 10)  # never wrap — gutter alignment survives
            if kind == "-":
                print(red(f"    - {num}│ ") + red(l))
            else:
                print(green(f"    + {num}│ ") + green(l))
            shown += 1
    if ws_only:
        print(dim(f"    (whitespace-only differences on {ws_only} line"
                  + ("s" if ws_only != 1 else "") + ")"))
    if overflow:
        print(dim("    (generation continued past function end — hidden, "
                  "'raw' shows it)"))


def _grade(code, gen):
    """Verdict ladder: BYTE-EXACT > CODE-EXACT > MISS > UNGRADEABLE.
    A MISS must mean 'the code genuinely differs' — when the ORIGINAL itself
    doesn't parse (so code_exact fail-closes for reasons that aren't the
    model's fault), the honest verdict is UNGRADEABLE, checked up front."""
    from compressor.exactness import normalize_code
    gradeable = normalize_code(code) is not None
    be = byte_exact(code, gen)
    ce = bool(gradeable and code_exact(code, gen))
    sim = difflib.SequenceMatcher(
        None, code, gen[: len(code) + 200], autojunk=False).ratio()
    return be, ce, sim, gradeable


def _verdict_tags(be, ce, sim, gradeable=True):
    if be:
        return boldgrn("✔ BYTE-EXACT"), "BYTE-EXACT"
    if ce:
        return yellow("~ CODE-EXACT"), "CODE-EXACT"
    if not gradeable:
        return _c("1;35", f"? UNGRADEABLE  sim {100 * sim:.1f}%"), "UNGRADEABLE"
    return boldred(f"✘ MISS  sim {100 * sim:.1f}%"), f"MISS {100 * sim:.0f}%"


_UNGRADEABLE_NOTE = ("    (the ORIGINAL doesn't parse as top-level Python, so "
                     "code-exact grading is impossible;\n     sim is raw "
                     "character similarity of the full text)")


def cmd_recon(args=None):
    if S["stack"]:
        stack_recon(" ".join(args) if args else None)
        return
    rec = need_rec()
    if rec is None:
        return
    tok = get_tok()
    code, name = rec["code"], rec["name"]
    cids = tok(code, add_special_tokens=False).input_ids
    budget = len(cids) + 16
    print(f"reconstructing {bold(name)} ({len(cids)} tokens, pf={S['pf']}, "
          f"arms: {' '.join(S['arms'])})")
    for arm in S["arms"]:
        t0 = time.perf_counter()
        try:
            with spinner(f"{arm} reconstructing"):
                dec, ctx, nctx = arm_context(arm, code, name, recon=True)
                out = batched_generate(dec, [ctx], [budget], batch_size=1)[0]
        except KeyboardInterrupt:
            print(f"\n{badge(arm)} interrupted — back to prompt")
            return
        dt = time.perf_counter() - t0
        gen = tok.decode(out.tolist(), skip_special_tokens=True)
        S["last_raw"][arm] = gen
        S["last_kind"] = "recon"
        be, ce, sim, gradeable = _grade(code, gen)
        if arm == "vec":
            rate = f"{len(cids) / max(nctx, 1):.1f}× ({nctx} vectors)"
        else:
            rate = f"1× ({nctx}-tok text)"
        tag, ptag = _verdict_tags(be, ce, sim, gradeable)
        # Print each arm's diff immediately so it can be read while the next
        # arm generates.
        print(f"{vpad(badge(arm), 8)} {tag}   {rate}   {dt:.1f}s")
        if not be and not gradeable:
            print(dim(_UNGRADEABLE_NOTE))
        if not be:
            print_diff(code, gen, ce)
    print(dim("('raw' shows the full generations side by side with the original)"))


def cmd_compare():
    """Show the vector arm's fidelity curve at pf 4, 6, 8, and 12."""
    rec = need_rec()
    if rec is None:
        return
    tok = get_tok()
    code, name = rec["code"], rec["name"]
    cids = tok(code, add_special_tokens=False).input_ids
    dec = get_ft_dec()
    comp = get_comp()
    print(f"rate sweep on {bold(name)} ({len(cids)} tokens) — {badge('vec')} "
          "arm at pf 4 / 6 / 8 / 12 (trained: 4 and 8)")
    results = []
    try:
        for pf in (4, 6, 8, 12):
            t0 = time.perf_counter()
            try:
                with spinner(f"pf{pf} reconstructing"):
                    comp.pooling_factor = pf
                    with torch.no_grad():
                        vecs = comp.compress(code, DEVICE)[0]
                    out = batched_generate(
                        dec, [vecs], [len(cids) + 16], batch_size=1)[0]
            except KeyboardInterrupt:
                print("\n(^C — sweep aborted)")
                return
            dt = time.perf_counter() - t0
            gen = tok.decode(out.tolist(), skip_special_tokens=True)
            be, ce, sim, gradeable = _grade(code, gen)
            tag, ptag = _verdict_tags(be, ce, sim, gradeable)
            nv = vecs.shape[0]
            print(f"  pf{pf:<3} {tag}   {len(cids) / nv:.1f}× ({nv} vectors)   {dt:.1f}s")
            results.append((pf, nv, be, ce, ptag, sim, dt))
    finally:
        comp.pooling_factor = S["pf"]
    print(dim("  ── rate vs fidelity ────────────────────────"))
    print(dim("    pf │ vectors │ verdict      │  sim% │  time"))
    for pf, nv, be, ce, ptag, sim, dt in results:
        col = boldgrn if be else yellow if ce else boldred
        print(f"    {pf:>2} │ {nv:>7} │ {col(f'{ptag:<12}')} │ {100 * sim:5.1f} │ "
              f"{dt:4.1f}s")
    print(dim("  (4 and 8 were trained; 6 interpolates; 10-12 is past the cliff)"))


def _short(s, n=36):
    s = s.strip() or "∅ (blank)"
    return s if len(s) <= n else s[: n - 1] + "…"


def cmd_sample(args):
    """N sampled reconstructions -> per-line agreement view. Greedy stays the
    published mode; sampling explores what the vectors pin down confidently."""
    rec = need_rec()
    if rec is None:
        return
    n, temp = 5, 0.8
    if args and args[0].replace(".", "").isdigit():
        n = max(2, min(16, int(float(args[0]))))
    if len(args) > 1:
        try:
            temp = float(args[1])
        except ValueError:
            pass
    tok = get_tok()
    code, name = rec["code"], rec["name"]
    arm = S["arms"][0]
    cids = tok(code, add_special_tokens=False).input_ids
    budget = len(cids) + 16
    print(f"sampling {n}× @ temp {temp} from {badge(arm)} (first active arm only "
          "to keep it fast)")
    print(dim("  greedy = the published-numbers mode; sampling shows where the "
              "context is confident vs uncertain"))
    try:
        with spinner("greedy reference"):
            dec, ctx, _ = arm_context(arm, code, name, recon=True)
            gout = batched_generate(dec, [ctx], [budget], batch_size=1)[0]
        greedy = tok.decode(gout.tolist(), skip_special_tokens=True)
        with spinner(f"{n} samples @ temp {temp}"):
            with torch.no_grad():
                emb = ctx.unsqueeze(0).expand(n, -1, -1).contiguous()
                mask = torch.ones(n, ctx.shape[0], dtype=torch.long, device=DEVICE)
                sout = dec.generate(inputs_embeds=emb, attention_mask=mask,
                                    max_new_tokens=budget, do_sample=True,
                                    temperature=temp, top_p=0.95,
                                    pad_token_id=tok.eos_token_id)
    except KeyboardInterrupt:
        print("\n(^C — back to prompt)")
        return
    samples = [tok.decode(s.tolist(), skip_special_tokens=True) for s in sout]
    # reference lines: the greedy reconstruction, trimmed to the function when
    # it reproduced it exactly (drops trailing budget overflow)
    glines = (code if greedy.startswith(code)
              else greedy[: len(code) + 200]).split("\n")
    L = len(glines)
    # align every sample to the greedy reference BEFORE voting, so an early
    # inserted/deleted line doesn't mark everything below it as disputed
    votes = [[] for _ in range(L)]
    for s in samples:
        slines = s[: len(code) + 200].split("\n")
        mapped = [None] * L
        for t_, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, glines, slines, autojunk=False).get_opcodes():
            if t_ == "equal":
                for k in range(i2 - i1):
                    mapped[i1 + k] = slines[j1 + k]
            elif t_ == "replace":
                for k in range(i2 - i1):
                    mapped[i1 + k] = slines[j1 + k] if j1 + k < j2 else None
        for i in range(L):
            votes[i].append(mapped[i])
    from collections import Counter
    hg = hl_lines("\n".join(glines))
    nw = max(2, len(str(L)))
    disputed = []
    n_unanimous = 0
    for i in range(L):
        cnt = Counter("∅ (line missing)" if v is None else v for v in votes[i])
        if len(cnt) == 1 and votes[i][0] == glines[i]:
            n_unanimous += 1
            print(dim(f"{i + 1:>{nw}}│ ") + hg[i])
            continue
        frac = 1 - cnt.most_common(1)[0][1] / n
        bg = "43;30" if frac <= 0.34 else "41;97"  # mild → yellow, heavy → red
        print(dim(f"{i + 1:>{nw}}│ ") + _c(bg, glines[i] or " "))
        disputed.append((i + 1, cnt))
    if disputed:
        print(dim("  disputed lines (votes across samples):"))
        for no, cnt in disputed[:12]:
            parts = " · ".join(f'{c}× "{_short(v)}"'
                               for v, c in cnt.most_common(3))
            more = "" if len(cnt) <= 3 else f" · +{len(cnt) - 3} more"
            print(f"    line {no}: {parts}{more}")
        if len(disputed) > 12:
            print(dim(f"    … {len(disputed) - 12} more disputed lines"))
    print(bold(f"{n} samples @ temp {temp}: {n_unanimous}/{L} lines unanimous, "
               f"{L - n_unanimous} disputed"))


def print_text_block(text, indent="  "):
    """Multi-line answer text, real newlines, blank-line runs collapsed."""
    blanks = 0
    for ln in text.split("\n"):
        if not ln.strip():
            blanks += 1
            continue
        if blanks:
            if blanks <= 2:
                print("\n" * (blanks - 1), end="")
                print()
            else:
                print(dim(f"{indent}⋮ ({blanks} blank lines)"))
            blanks = 0
        print(indent + ln)
    if blanks > 2:
        print(dim(f"{indent}⋮ ({blanks} blank lines)"))


def cmd_ask(question):
    if S["stack"]:
        stack_ask(question)
        return
    rec = need_rec()
    if rec is None:
        return
    tok = get_tok()
    code, name, qa = rec["code"], rec["name"], rec["qa"]
    gold = None
    if question:
        prompt = f"\nQ: {question}\nA:"
    elif qa:
        get_qa()  # ensures qa32 module is loaded
        prompt = _m["qa32"].qa_prompt(qa)  # letters the options iff MCQ
        question = qa["question"]
        gold = qa["answer"]
    else:
        print("no gold question on this record — use: ask <your question>")
        return
    pids = tok(prompt, add_special_tokens=False).input_ids
    print(f"Q: {question}")
    if gold is not None:
        print(f"gold A: {bold(gold)}")
    for arm in S["arms"]:
        t0 = time.perf_counter()
        try:
            with spinner(f"{arm} answering"):
                dec, ctx, nctx = arm_context(arm, code, name, prompt_ids=pids)
                out = batched_generate(dec, [ctx], [200], batch_size=1)[0]
        except KeyboardInterrupt:
            print(f"\n{badge(arm)} interrupted — back to prompt")
            return
        dt = time.perf_counter() - t0
        gen = tok.decode(out.tolist(), skip_special_tokens=True)
        S["last_raw"][arm] = gen
        S["last_kind"] = "ask"
        graded = ""
        if gold is not None and qa.get("answer_type"):
            from dataset.qa_filters import rt_grade
            first = gen.split("\n", 1)[0].strip()[:200]
            ok = rt_grade(first, qa)
            graded = "  [grader on first line: " + \
                (boldgrn("PASS") if ok else boldred("FAIL")) + "]"
        print(f"--- {badge(arm)} ({dt:.1f}s, ctx {nctx}){graded} ---")
        print_text_block(gen)


# -------------------------------------------------------------- raw rendering
def _raw_rows(a, b):
    """Line-paired diff rows for raw display: (mark, a_idx|None, b_idx|None).
    Overflow (generation past the original's end) is returned separately."""
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    j_end, last = 0, None
    for op in ops:
        if op[2] > op[1]:  # consumes original lines
            last = op
    if last:
        t, i1, i2, j1, j2 = last
        # pair-align the final consuming opcode: surplus generated lines
        # beyond the last original line belong to the overflow section
        j_end = min(j2, j1 + (i2 - i1)) if t == "replace" else j2
    rows = []
    for t, i1, i2, j1, j2 in ops:
        if t == "equal":
            rows += [(" ", i1 + k, j1 + k) for k in range(i2 - i1)]
        elif t == "delete":
            rows += [("-", i1 + k, None) for k in range(i2 - i1)]
        elif t == "insert":
            rows += [("+", None, j1 + k) for k in range(j2 - j1) if j1 + k < j_end]
        else:  # replace — pair index-wise
            for k in range(max(i2 - i1, j2 - j1)):
                ai = i1 + k if i1 + k < i2 else None
                bj = j1 + k if j1 + k < j2 else None
                if bj is not None and bj >= j_end:
                    bj = None
                if ai is None and bj is None:
                    continue
                mark = ("~" if ai is not None and bj is not None
                        else "-" if bj is None else "+")
                rows.append((mark, ai, bj))
    return rows, j_end


_MARKC = {"-": red, "+": green, "~": yellow, " ": lambda s: s}


def render_raw_recon(arm, code, gen):
    """ORIGINAL vs generation, syntax highlighted, line-numbered. Side by side
    on wide terminals; a single git-style listing on narrow ones. Changed
    lines are shown as red-original/green-generated pairs with the exact
    differing character spans background-highlighted. Overflow past the
    function's end is shown dimmed after a separator — raw means raw."""
    a, b = code.split("\n"), gen.split("\n")
    ha, hb = hl_lines(code), hl_lines(gen)
    rows, j_end = _raw_rows(a, b)
    nw = max(2, len(str(max(len(a), len(b)))))
    width = shutil.get_terminal_size().columns

    def num(n):
        return dim(f"{n:>{nw}}│") if n is not None else " " * nw + "│"

    if width >= 100:  # side-by-side panels
        colw = (width - 2 * (nw + 1) - 5) // 2
        print(bold(f"  {' ' * (nw + 1)}{vpad('ORIGINAL', colw)}"
                   f"│ {' ' * (nw + 1)}{arm} generation"))
        for mark, ai, bj in rows:
            if mark == "~":
                l, r = _pair_hl(a[ai], b[bj])
            else:
                l = ha[ai] if ai is not None else ""
                r = hb[bj] if bj is not None else ""
                if mark == "-":
                    l = red(a[ai])
                elif mark == "+":
                    r = green(b[bj])
            print(f"{_MARKC[mark](mark)} "
                  f"{num(ai + 1 if ai is not None else None)}"
                  f"{vpad(vtrunc(l, colw), colw)}│ "
                  f"{num(bj + 1 if bj is not None else None)}{vtrunc(r, colw)}")
        if j_end < len(b):
            print(dim("  ── generation continues past function end ──"))
            for j in range(j_end, len(b)):
                print(dim(f"  {' ' * nw}│ " + clipw(ANSI_RE.sub("", hb[j]), nw + 5)))
    else:  # narrow: single git-style listing (clipped — never wraps)
        res = nw + 5
        print(bold(f"── {arm} generation vs ORIGINAL ({len(a)} lines) ──"))
        for mark, ai, bj in rows:
            if mark == " ":
                print(f"  {num(ai + 1)} " + clipw(hb[bj], res))
            elif mark == "~":
                l, r = _pair_hl(a[ai], b[bj])
                print(f"{red('-')} {num(ai + 1)} " + clipw(l, res))
                print(f"{green('+')} {num(bj + 1)} " + clipw(r, res))
            elif mark == "-":
                print(f"{red('-')} {num(ai + 1)} " + red(clipw(a[ai], res)))
            else:
                print(f"{green('+')} {num(bj + 1)} " + green(clipw(b[bj], res)))
        if j_end < len(b):
            print(dim("  ── generation continues past function end ──"))
            for j in range(j_end, len(b)):
                print(dim(f"  {' ' * nw}│ " + clipw(ANSI_RE.sub("", hb[j]), nw + 5)))


def cmd_raw():
    if not S["last_raw"]:
        print("nothing generated yet")
        return
    rec = S["rec"]
    if S["last_kind"] == "recon" and rec:
        for arm, gen in S["last_raw"].items():
            render_raw_recon(arm, rec["code"], gen)
    else:
        for arm, gen in S["last_raw"].items():
            print(bold("── ") + badge(arm) + bold(" full answer ──"))
            print_text_block(gen)


def cmd_gold():
    rec = need_rec()
    if rec is None:
        return
    if not rec["qa"]:
        print("this record has no gold QA (only 'load qa <idx>' records do)")
        return
    qa = rec["qa"]
    print(f"intent: {qa['intent']}   answer_type: {qa.get('answer_type')}")
    print(f"Q: {qa['question']}")
    if qa.get("options"):
        for i, o in enumerate(qa["options"]):
            print(f"  {chr(65 + i)}) {o}")
    print(f"A: {bold(qa['answer'])}")


def cmd_show():
    rec = need_rec()
    if rec is None:
        return
    print(f"[{rec['src']}] {bold(rec['name'])}")
    lines = hl_lines(rec["code"])
    nw = max(2, len(str(len(lines))))
    for i, l in enumerate(lines, start=1):
        print(dim(f"{i:>{nw}}│ ") + clipw(l, nw + 3))


def cmd_demo():
    """Canned ~60s showcase with narration."""
    print(dim("demo — 60 seconds, 3 stages."))
    print(dim("the system: an encoder maps a function into about pf× fewer decoder"))
    print(dim("context slots; a finetuned decoder must rebuild the exact source"))
    print(dim("— or answer questions — from those continuous vectors alone."))
    old_arms, old_pf = list(S["arms"]), S["pf"]
    S["pf"] = 4
    try:
        print()
        print(dim("stage 1 · load a canary function (from held-out repositories):"))
        cmd_load(["canary", "5"])
        n = S["rec"]["n_tok"]
        print()
        print(dim(f"stage 2 · map {n} tokens into ~{math.ceil(n / 4)} vectors "
                  f"(pf=4 → about four times fewer decoder context slots) and"))
        print(dim(f"         reconstruct from vectors alone — arm {ANSI_RE.sub('', badge('vec'))}, "
                  "no text prompt, no name:"))
        S["arms"] = ["vec"]
        cmd_recon()
        print(dim("         BYTE-EXACT = every byte recovered from a representation"))
        print(dim("         using about four times fewer decoder context slots."))
        print()
        print(dim("stage 3 · the vectors aren't a text copy; the decoder can ANSWER from them:"))
        cmd_ask("what does this function yield?")
        print()
        print(dim(f"done. arms: {ANSI_RE.sub('', badge('vec'))}=vectors  "
                  f"{ANSI_RE.sub('', badge('text'))}=same model reading raw code  "
                  f"{ANSI_RE.sub('', badge('stock'))}=untouched Qwen3-1.7B"))
        print(dim("try: paste — feed it your own code · compare — the rate cliff"))
    finally:
        S["arms"], S["pf"] = old_arms, old_pf


HELP = """commands:
  demo                     60-second guided tour — start here
  load canary <idx|name>   load one of the 36 canary functions (0-35, or name part)
  load ood <idx>           load one of the 600 post-cutoff OOD functions (0-599)
  load qa <idx>            load a fully OOD QA record (has a gold question+answer)
  paste                    paste your own function; end with a line that is just .
  show                     print the currently loaded function (highlighted, numbered)
  pf <n>                   set pooling factor (default 4; 8 is also trained)
  arms <list>              choose arms, e.g.: arms vec text   (default: vec text stock)
  recon [name|idx]         reconstruct from each arm; grades exactness. With a
                           stack loaded, name/index picks the TARGET member
  stack add <what>         build a multi-function stack: canary <i>|ood <i>|qa <i>|paste
  stack show|rm|clear      stack management
  compare                  rate sweep pf 4/6/8/12 on the loaded function (vec arm)
  sample [N] [temp]        N sampled reconstructions (default 5 @ 0.8) -> per-line
                           agreement view: where the vectors are confident vs not
  ask <question>           ask a question about the function (omit it on a qa record
                           to use the gold question)
  gold                     show the loaded qa record's gold question/answer
  raw                      full generations: recon -> side-by-side vs the original
                           with char-level diff highlights; ask -> full answers
  help                     this text
  quit                     exit
arms: [vec] = finetuned model reading vectors in about pf-times fewer context slots
      [text] = same finetuned model reading the raw code text
      [stock] = untouched Qwen3-1.7B reading the raw code text"""


def _status():
    if S["stack"]:
        st = S["stack"]
        tot, vec = _stack_est(st["members"])
        return dim(f"[stack: {len(st['members'])} fns · {tot} tok → {vec} vec · "
                   f"pf{S['pf']} · arms: {' '.join(S['arms'])}]")
    rec = S["rec"]
    where = (f"{rec['name']} · {rec['n_tok']} tok" if rec else "no function")
    return dim(f"[{where} · pf{S['pf']} · arms: {' '.join(S['arms'])}]")


def main():
    global _LOG
    # session transcript: plain text, ANSI stripped, for quoting later
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"session_{time.strftime('%Y%m%d_%H%M%S')}.log"
    _LOG = open(log_path, "w")
    sys.stdout = _Tee(_REAL_STDOUT, _LOG)

    # arrow-key history + line editing (macOS ships libedit-backed readline).
    # NOTE: the input() prompt must stay 100% plain — ANSI escapes in a prompt
    # confuse libedit's cursor math; the colored status line is printed
    # separately above it.
    try:
        import readline
        hist = Path.home() / ".playground_history"
        try:
            readline.read_history_file(hist)
        except Exception:
            pass
        import atexit
        atexit.register(lambda: _safe_write_history(readline, hist))
    except Exception:
        pass  # REPL works fine without line editing

    if ARGS.arms:
        sel = [a.strip() for a in ARGS.arms.split(",") if a.strip()]
        if all(a in ALL_ARMS for a in sel) and sel:
            S["arms"] = [a for a in ALL_ARMS if a in sel]
    if 1 <= ARGS.pf <= 64:
        S["pf"] = ARGS.pf

    print(bold("exact-latent playground") + " — released checkpoint")
    print(f"  weights: {MODEL_STATE.relative_to(ROOT)} (+ {PROJECTOR_EMA.name})")
    print(f"  device: {DEVICE_DESC} | arms: {' '.join(S['arms'])} | pf: {S['pf']}")
    print(f"  transcript: {log_path.relative_to(ROOT)}")
    print(dim("new here? type: demo   (or: load canary 5 · paste · help)"))

    # Bracketed paste is deliberately disabled: some line editors can wedge
    # while parsing a marker that arrives with a fast multi-line paste.
    # Without the mode, terminals paste plain text: libedit sees only the
    # first line, and _paste_event's shape/pending heuristics route the rest
    # into raw capture. Marker stripping stays as defense for terminals that
    # send markers anyway.
    try:
        _main_loop()
    finally:
        pass
    print("bye")
    print(dim(f"(transcript saved: {log_path})"))


def _main_loop():
    while True:
        try:
            print(_status())
            raw_line = input("pg> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n(^C — type 'quit' to exit)")
            continue
        if raw_line and _paste_event(raw_line):
            # a paste landed at the prompt — capture it as code instead of
            # firing every pasted line as a bogus command
            print("(detected multi-line paste — capturing as code; end with .)")
            _log_line("pg> (paste event)")
            _finalize_paste(_capture_paste(first_line=raw_line, explicit=False),
                            auto=True)
            continue
        line = raw_line.strip()
        if not line:
            continue
        _log_line(f"pg> {line}")
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]
        try:
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "demo":
                cmd_demo()
            elif cmd == "load":
                cmd_load(args)
            elif cmd == "paste":
                cmd_paste()
            elif cmd == "pf":
                if args and args[0].isdigit() and 1 <= int(args[0]) <= 64:
                    S["pf"] = int(args[0])
                    trained = "trained rate" if S["pf"] in (4, 8) else \
                        "NEVER-trained rate (interpolates ok to ~6, cliff past 10)"
                    print(f"pf = {S['pf']} ({trained})")
                else:
                    print(f"usage: pf <1-64>   (currently {S['pf']})")
            elif cmd == "arms":
                sel = [a for a in args if a in ALL_ARMS]
                if not sel or len(sel) != len(args):
                    print(f"usage: arms <any of: {' '.join(ALL_ARMS)}>  "
                          f"(currently: {' '.join(S['arms'])})")
                else:
                    S["arms"] = [a for a in ALL_ARMS if a in sel]  # stable order
                    print(f"arms: {' '.join(badge(a) for a in S['arms'])}")
            elif cmd == "recon":
                cmd_recon(args)
            elif cmd == "stack":
                cmd_stack(args)
            elif cmd == "compare":
                cmd_compare()
            elif cmd == "sample":
                cmd_sample(args)
            elif cmd == "ask":
                cmd_ask(" ".join(args) if args else None)
            elif cmd == "gold":
                cmd_gold()
            elif cmd == "raw":
                cmd_raw()
            elif cmd == "show":
                cmd_show()
            else:
                print(f"unknown command '{cmd}' — type 'help'")
        except KeyboardInterrupt:
            print("\n(^C — back to prompt)")
        except PgError as e:
            print(boldred("problem: ") + str(e))
        except Exception as e:  # keep the REPL alive on any error
            print(f"error: {type(e).__name__}: {e}")


def _safe_write_history(readline_mod, path):
    try:
        readline_mod.write_history_file(path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
