"""Corpus sampler stage B: deduplicate, filter, and draw stratified splits.

Pure function of the annot parquets (dataset/annotate.py) + SEED. No code
text is read here; the output is a draw manifest (which exact rows go where)
plus a rejects manifest (every dropped row with its reason) plus stats.json
with every number the VALIDATION_REPORT needs.

Order of precedence for drop reasons (each row gets exactly one status):
  excl_hash / excl_repo  -> independent eval-exclusion recheck (expected 0 hits)
  dup                    -> md5(code.strip()) seen earlier in canonical order
                            (canonical order = stackv3, starcoder, starcoder_long;
                             shard, then line — scarce modern pile wins ties)
  junk                   -> 1 hard or >=2 soft minification trips
  family_cap             -> boilerplate family (>=10 distinct repos) already has
                            20 kept examples / repo already used in that family
  stub_cap               -> stub beyond 1% of the split target
  repo_cap               -> repo already has 150 kept rows across the final corpus
  val_reserved           -> row lives in a validation-reserved repo, not drawn
  not_drawn              -> eligible, band already at target (surplus supply)

Draw policy (fixed bands; stackv3-first):
  - validation repos reserved FIRST from starcoder piles (disjoint from train)
  - ALL eligible stackv3 rows -> train (subject to caps only)
  - remainder per length band filled from starcoder + starcoder_long
  - band targets over the 2.6M train corpus: 16-63:10% 64-255:60% 256-511:20%
    512-1023:8% 1024-4095:2%; shortfalls reported, never rebalanced silently.

Run: uv run python dataset/draw.py
"""
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CV2 = ROOT / "data" / "corpus_v2"
META = CV2 / "corpus_meta"
ANNOT = META / "annot"

SEED = 1807
TRAIN_TARGET = 2_600_000
VAL_TARGET = 5_000
BANDS = [(16, 63), (64, 255), (256, 511), (512, 1023), (1024, 4095)]
BAND_SHARE = [0.10, 0.60, 0.20, 0.08, 0.02]
TRAIN_BAND_TARGET = [int(TRAIN_TARGET * s) for s in BAND_SHARE]     # sums to 2.6M
VAL_BAND_TARGET = [int(VAL_TARGET * s) for s in BAND_SHARE]         # sums to 5000
FAMILY_REPO_MIN = 10      # family spanning >= this many distinct repos = boilerplate
FAMILY_CAP = 20           # kept examples per boilerplate family, distinct repos
REPO_CAP = 150            # global across final corpus (train+val)
STUB_SHARE = 0.01
PILES = {0: "stackv3", 1: "starcoder", 2: "starcoder_long", 3: "starcoder_valley"}

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


# --- eval exclusion refs, built exactly as dataset/corpus_build.py does -------
excl_hashes, excl_repos = set(), set()
for f in ("canaries.jsonl", "canaries_v2.jsonl", "canaries_ood.jsonl", "canaries_smoke.jsonl"):
    for l in open(ROOT / "data" / f):
        excl_hashes.add(hashlib.md5(json.loads(l)["code"].strip().encode()).hexdigest())
for l in open(ROOT / "data/postcut_corpus/functions.jsonl"):
    r = json.loads(l)
    excl_hashes.add(hashlib.md5(r["code"].strip().encode()).hexdigest())
    excl_repos.add(r["repo"].lower())
for l in open(ROOT / "data/postcut_corpus/manifest.jsonl"):
    excl_repos.add(json.loads(l)["repo"].lower())
assert len(excl_hashes) > 25_000 and len(excl_repos) > 200, \
    f"eval exclusion refs look wrong ({len(excl_hashes)} hashes, {len(excl_repos)} repos) — refusing to run"
log(f"exclusion refs: {len(excl_hashes)} hashes, {len(excl_repos)} repos")
excl_hash_b = {bytes.fromhex(h) for h in excl_hashes}

# --- load annot metadata ------------------------------------------------------
parts = sorted(ANNOT.glob("p*_s*.parquet"))
log(f"loading {len(parts)} annot parquets ...")
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
df.sort_values(["pile", "shard", "line"], inplace=True, kind="mergesort")
df.reset_index(drop=True, inplace=True)
N = len(df)
log(f"pool: {N} rows " + str(df.groupby("pile").size().to_dict()))

stats = {"seed": SEED, "pool_total": N,
         "pool_per_pile": {PILES[k]: int(v) for k, v in df.groupby("pile").size().items()},
         "tokenize_fallback": int(df.fam_fb.sum())}

# band index (-1 = out of every band; harvest guaranteed 16..4096 so expect none)
nt = df.n_tokens.to_numpy()
band = np.full(N, -1, dtype=np.int8)
for i, (lo, hi) in enumerate(BANDS):
    band[(nt >= lo) & (nt <= hi)] = i
# harvest allowed exactly 4096; band 4 spec is 1024-4095 -> 4096-token rows
# fold into band 4 (report the count rather than orphaning them)
n4096 = int((nt == 4096).sum())
band[nt == 4096] = 4
df["band"] = band
stats["rows_at_4096_folded_into_band4"] = n4096
assert int((band < 0).sum()) == 0, "rows outside 16..4096 — harvest contract broken"

status = np.full(N, "", dtype=object)   # final status per row

# --- independent eval-exclusion recheck (hash AND repo) on ALL rows -----------
h_arr = df.h.to_numpy()
repo_arr = df.repo.str.lower().to_numpy()
hit_hash = np.fromiter((h in excl_hash_b for h in h_arr), bool, N)
hit_repo = np.fromiter((r in excl_repos for r in repo_arr), bool, N)
stats["exclusion_checked_rows"] = N
stats["exclusion_hash_hits"] = int(hit_hash.sum())
stats["exclusion_repo_hits"] = int(hit_repo.sum())
status[hit_hash] = "excl_hash"
status[hit_repo & ~hit_hash] = "excl_repo"
log(f"exclusion recheck: {N} rows checked, {int(hit_hash.sum())} hash hits, "
    f"{int(hit_repo.sum())} repo hits (expected 0/0)")

# --- cross-pile content dedup (first-seen wins in canonical order) ------------
dup_mask = df.h.duplicated(keep="first").to_numpy().copy()
dup_mask = dup_mask & (status == "")
status[dup_mask] = "dup"
per_pile_dup = {PILES[k]: int(v) for k, v in
                df.pile[dup_mask].value_counts().sort_index().items()}
stats["dedup_losses_per_pile"] = per_pile_dup
log(f"dedup: {int(dup_mask.sum())} rows dropped {per_pile_dup}")

# --- junk ---------------------------------------------------------------------
junk_mask = df.junk.to_numpy() & (status == "")
status[junk_mask] = "junk"
stats["junk_rule_counts"] = {   # trips counted over the deduped pool (pre-junk-drop)
    "hard_max_line_gt_1000": int(df.hard_line[status != "dup"].sum()),
    "soft_mean_line_gt_200": int(df.soft_mean[status != "dup"].sum()),
    "soft_ws_ratio_lt_5pct": int(df.soft_ws[status != "dup"].sum()),
    "soft_alnum_run_gt_200": int(df.soft_run[status != "dup"].sum()),
    "soft_string_dominant": int(df.soft_str[status != "dup"].sum()),
}
stats["junk_dropped"] = int(junk_mask.sum())
log(f"junk: {int(junk_mask.sum())} dropped; rule trips {stats['junk_rule_counts']}")

eligible = status == ""
stats["eligible_pool"] = int(eligible.sum())
stats["eligible_per_pile"] = {PILES[k]: int(v) for k, v in
                              df.pile[eligible].value_counts().sort_index().items()}

# --- family stats over the eligible pool --------------------------------------
log("family stats ...")
fam_arr = df.fam.to_numpy()
fam_codes, fam_inv = np.unique(fam_arr[eligible], return_inverse=True)
elig_idx = np.flatnonzero(eligible)
fam_size = np.bincount(fam_inv)
# distinct repos per family
fam_repos = defaultdict(set)
rl = repo_arr[eligible]
for j, fi in enumerate(fam_inv):
    if fam_size[fi] >= 2:
        fam_repos[fi].add(rl[j])
fam_nrepo = {fi: len(s) for fi, s in fam_repos.items()}
boiler = {fi for fi, nr in fam_nrepo.items() if nr >= FAMILY_REPO_MIN}
stats["families_total"] = int(len(fam_codes))
stats["families_boilerplate"] = len(boiler)
stats["rows_in_boilerplate_families"] = int(sum(fam_size[fi] for fi in boiler))
log(f"families: {len(fam_codes)} total, {len(boiler)} boilerplate "
    f"({stats['rows_in_boilerplate_families']} rows)")

# map row -> family index (only need it for eligible rows)
row_fam = np.full(N, -1, dtype=np.int64)
row_fam[elig_idx] = fam_inv

# --- draw ---------------------------------------------------------------------
rng = np.random.default_rng(SEED)
repo_kept = Counter()          # global repo cap
fam_kept = Counter()           # boilerplate family cap
fam_repo_used = defaultdict(set)
stub_arr = df.stub.to_numpy()
pile_arr = df.pile.to_numpy()
band_arr = df.band.to_numpy()

TRAIN_STUB_CAP = int(TRAIN_TARGET * STUB_SHARE)
VAL_STUB_CAP = int(VAL_TARGET * STUB_SHARE)


def try_take(i, stub_cap, stub_count):
    """Return (ok, reason). Mutates cap counters on accept."""
    r = repo_arr[i]
    if repo_kept[r] >= REPO_CAP:
        return False, "repo_cap"
    fi = row_fam[i]
    if fi in boiler:
        if fam_kept[fi] >= FAMILY_CAP or r in fam_repo_used[fi]:
            return False, "family_cap"
    if stub_arr[i] and stub_count[0] >= stub_cap:
        return False, "stub_cap"
    repo_kept[r] += 1
    if fi in boiler:
        fam_kept[fi] += 1
        fam_repo_used[fi].add(r)
    if stub_arr[i]:
        stub_count[0] += 1
    return True, ""


# --- validation repo reservation (starcoder piles only; stackv3 is train-only)
log("reserving validation repos ...")
sc_elig = eligible & (pile_arr != 0)
repo_band_counts = defaultdict(lambda: np.zeros(5, dtype=np.int64))
for i in np.flatnonzero(sc_elig):
    repo_band_counts[repo_arr[i]][band_arr[i]] += 1
all_sc_repos = np.array(sorted(repo_band_counts.keys()))
rng.shuffle(all_sc_repos)
need = np.array(VAL_BAND_TARGET) * 2       # 2x margin for cap rejections
acc = np.zeros(5, dtype=np.int64)
val_repos = set()
for r in all_sc_repos:
    if (acc >= need).all():
        break
    val_repos.add(r)
    acc += repo_band_counts[r]
stats["val_reserved_repos"] = len(val_repos)
log(f"reserved {len(val_repos)} repos for validation (band supply {acc.tolist()})")

# --- validation draw ----------------------------------------------------------
val_stub = [0]
val_taken = [0] * 5
in_val_repo = np.fromiter((r in val_repos for r in repo_arr), bool, N)
val_cand = np.flatnonzero(eligible & in_val_repo & (pile_arr != 0))
rng.shuffle(val_cand)
for i in val_cand:
    b = band_arr[i]
    if val_taken[b] >= VAL_BAND_TARGET[b]:
        continue
    ok, why = try_take(i, VAL_STUB_CAP, val_stub)
    if ok:
        status[i] = "kept_val"
        val_taken[b] += 1
    else:
        status[i] = why
# leftover rows in val repos are quarantined from train (disjointness)
leftover = eligible & in_val_repo & (status == "")
status[leftover] = "val_reserved"
stats["val_drawn_per_band"] = val_taken
stats["val_stubs"] = val_stub[0]
stats["val_reserved_not_drawn"] = int(leftover.sum())
log(f"validation: {sum(val_taken)}/{VAL_TARGET} drawn per-band {val_taken}, "
    f"{int(leftover.sum())} quarantined leftovers")

# --- train phase 1: all eligible stackv3 --------------------------------------
train_stub = [0]
train_taken = np.zeros(5, dtype=np.int64)
sv3_cand = np.flatnonzero(eligible & (pile_arr == 0) & (status == ""))
rng.shuffle(sv3_cand)
for i in sv3_cand:
    ok, why = try_take(i, TRAIN_STUB_CAP, train_stub)
    if ok:
        status[i] = "kept_train"
        train_taken[band_arr[i]] += 1
    else:
        status[i] = why
stats["train_stackv3_per_band"] = train_taken.tolist()
log(f"stackv3 phase: {int(train_taken.sum())} rows in, per-band {train_taken.tolist()}")

# --- train phase 2: fill bands from starcoder piles ---------------------------
for b in range(5):
    target = TRAIN_BAND_TARGET[b]
    cand = np.flatnonzero(eligible & (pile_arr != 0) & (status == "") & (band_arr == b))
    rng.shuffle(cand)
    for i in cand:
        if train_taken[b] >= target:
            break
        ok, why = try_take(i, TRAIN_STUB_CAP, train_stub)
        if ok:
            status[i] = "kept_train"
            train_taken[b] += 1
        else:
            status[i] = why
    if train_taken[b] < target:
        log(f"*** BAND SHORTFALL band{b} {BANDS[b]}: {int(train_taken[b])}/{target}")
log(f"train: {int(train_taken.sum())}/{TRAIN_TARGET} per-band {train_taken.tolist()}")

status[status == ""] = "not_drawn"
stats["train_per_band"] = train_taken.tolist()
stats["train_band_target"] = TRAIN_BAND_TARGET
stats["train_total"] = int((status == "kept_train").sum())
stats["val_total"] = int((status == "kept_val").sum())
stats["train_stubs"] = train_stub[0]
stats["status_counts"] = {k: int(v) for k, v in Counter(status).items()}

# per-band per-pile composition of the final train corpus
kept_tr = status == "kept_train"
comp = {}
for b in range(5):
    m = kept_tr & (band_arr == b)
    comp[f"band{b}"] = {PILES[k]: int(v) for k, v in
                        df.pile[m].value_counts().sort_index().items()}
stats["train_band_pile_composition"] = comp

# style axes per band (train), per pile
style_cols = ["decorated", "typed", "has_docstring", "is_method", "is_async"]
sty = {}
for b in range(5):
    m = kept_tr & (band_arr == b)
    d = {c: round(float(df[c][m].mean()), 4) for c in style_cols}
    d["string_share_mean"] = round(float(df.string_share[m].mean()), 4)
    d["n"] = int(m.sum())
    sty[f"band{b}"] = d
stats["train_style_per_band"] = sty
sty_p = {}
for p in (0, 1, 2):
    m = kept_tr & (pile_arr == p)
    d = {c: round(float(df[c][m].mean()), 4) for c in style_cols}
    d["string_share_mean"] = round(float(df.string_share[m].mean()), 4)
    d["n"] = int(m.sum())
    sty_p[PILES[p]] = d
stats["train_style_per_pile"] = sty_p

# stub share, timestamp clamps, modern share
stats["ts_clamped_future"] = int(df.ts_clamped.sum())
sv3 = pile_arr == 0
ts = df.file_ts.to_numpy()
POST = 1719792000  # 2024-07-01
stats["stackv3_post_2024_07_share"] = round(
    float(((ts >= POST) & sv3).sum() / max(sv3.sum(), 1)), 4)
kept_sv3 = kept_tr & sv3
stats["train_stackv3_post_2024_07_share"] = round(
    float(((ts >= POST) & kept_sv3).sum() / max(kept_sv3.sum(), 1)), 4)

# repo concentration over final train corpus
rk = pd.Series(repo_arr[kept_tr]).value_counts()
stats["train_repos"] = int(len(rk))
stats["train_top30_repos"] = {k: int(v) for k, v in rk.head(30).items()}
top1pct = max(1, len(rk) // 100)
stats["train_top1pct_repo_share"] = round(float(rk.head(top1pct).sum() / rk.sum()), 4)
v = rk.to_numpy()[::-1].astype(float)  # ascending
cum = np.cumsum(v)
gini = 1 - 2 * float((cum / cum[-1]).mean()) + 1 / len(v)
stats["train_repo_gini"] = round(abs(gini), 4)

# validation/train repo overlap (should be 0 by construction)
tr_repos = set(repo_arr[kept_tr])
va_repos = set(repo_arr[status == "kept_val"])
stats["train_val_repo_overlap"] = len(tr_repos & va_repos)

# --- top-50 families table -----------------------------------------------------
order = np.argsort(fam_size)[::-1][:50]
fam_rows = []
kept_mask_any = (status == "kept_train") | (status == "kept_val")
for fi in order:
    members = elig_idx[fam_inv == fi]
    kept_n = int(kept_mask_any[members].sum())
    ex = int(members[0])
    fam_rows.append(dict(
        fam=fam_codes[fi].hex(), size=int(fam_size[fi]),
        n_repos=int(fam_nrepo.get(fi, 1)), boilerplate=bool(fi in boiler),
        kept=kept_n, dropped=int(fam_size[fi]) - kept_n,
        example_id=f"p{pile_arr[ex]}:s{df.shard.iat[ex]}:l{df.line.iat[ex]}"))
json.dump(fam_rows, open(META / "families_top50.json", "w"), indent=1)

# --- persist manifests ---------------------------------------------------------
log("writing manifests ...")
df_out = df[["pile", "shard", "line", "band", "n_tokens"]].copy()
df_out["status"] = status
df_out.to_parquet(META / "draw_manifest.parquet", index=False)

# rejects manifest: every non-kept row (id, pile, reason) — no code text
import gzip
with gzip.open(META / "rejects_manifest.jsonl.gz", "wt") as f:
    p_, s_, l_ = df.pile.to_numpy(), df.shard.to_numpy(), df.line.to_numpy()
    for i in np.flatnonzero(~kept_mask_any):
        f.write(json.dumps({"id": f"p{p_[i]}:s{s_[i]}:l{l_[i]}",
                            "pile": PILES[p_[i]], "reason": status[i]}) + "\n")

# samples for materialize: junk dump, eyeball, surprisal-5k
junk_ids = np.flatnonzero(status == "junk")
sel = rng.choice(junk_ids, size=min(20, len(junk_ids)), replace=False)
json.dump([f"p{p_[i]}:s{s_[i]}:l{l_[i]}" for i in sel],
          open(META / "junk_sample_ids.json", "w"))

tr_ids = np.flatnonzero(kept_tr)
eye = rng.choice(tr_ids, size=150, replace=False)
json.dump([f"p{p_[i]}:s{s_[i]}:l{l_[i]}" for i in eye],
          open(META / "eyeball_ids.json", "w"))

surp_ids = []
for b in range(5):
    bi = np.flatnonzero(kept_tr & (band_arr == b))
    take = min(VAL_BAND_TARGET[b], len(bi))  # same 10/60/20/8/2 shape as val
    surp_ids += [int(x) for x in rng.choice(bi, size=take, replace=False)]
json.dump([f"p{p_[i]}:s{s_[i]}:l{l_[i]}" for i in surp_ids],
          open(META / "surprisal_ids.json", "w"))

# round-trip sample: 1000 random kept train rows
rt = rng.choice(tr_ids, size=1000, replace=False)
json.dump([f"p{p_[i]}:s{s_[i]}:l{l_[i]}" for i in rt],
          open(META / "roundtrip_ids.json", "w"))

json.dump(stats, open(META / "draw_stats.json", "w"), indent=1)
log("draw complete")
print(json.dumps({k: v for k, v in stats.items()
                  if not isinstance(v, dict) or len(str(v)) < 300}, indent=1))
