"""Modern-slice repo discovery (2023 -> 2025-08) via gh search, READ-ONLY.

Enumerates permissively-licensed Python repos by creation-date windows and
star bands (search caps at 1000 results/query — windows keep each query under
that). Output: data/corpus_v2/modern_repos.jsonl (full_name, created, stars,
license, clone_url). Politely paced for the 30 req/min search limit; resumable
(skips windows already in the output file).

Run: uv run python dataset/gh_modern_discover.py
"""
import json
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/corpus_v2/modern_repos.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

LICENSES = ["mit", "apache-2.0", "bsd-3-clause"]
STAR_BANDS = ["10..50", "51..300", ">300"]
WINDOW_DAYS = 21
START, END = date(2023, 1, 1), date(2025, 8, 31)

done_windows = set()
seen_repos = set()
if OUT.exists():
    for l in open(OUT):
        r = json.loads(l)
        done_windows.add(r.get("_window", ""))
        seen_repos.add(r["full_name"])
    print(f"resume: {len(seen_repos)} repos, {len(done_windows)} window-cells done")

out_f = open(OUT, "a")
n_new = 0
d = START
while d <= END:
    d2 = min(d + timedelta(days=WINDOW_DAYS - 1), END)
    for lic in LICENSES:
        for band in STAR_BANDS:
            wkey = f"{d}:{lic}:{band}"
            if wkey in done_windows:
                continue
            q = (f"search/repositories?q=language:python+created:{d}..{d2}"
                 f"+stars:{band}+license:{lic}&per_page=100&sort=stars")
            got = 0
            for page in (1, 2, 3):  # up to 300/cell; star-sorted so we keep the best
                try:
                    res = subprocess.run(
                        ["gh", "api", f"{q}&page={page}"],
                        capture_output=True, text=True, timeout=60)
                    if res.returncode != 0:
                        if "rate limit" in res.stderr.lower():
                            print("search rate-limited, sleeping 70s", flush=True)
                            time.sleep(70)
                            continue
                        break
                    items = json.loads(res.stdout).get("items", [])
                except Exception as e:
                    print(f"  {wkey} p{page}: {e}", flush=True)
                    break
                for it in items:
                    fn = it["full_name"]
                    if fn in seen_repos:
                        continue
                    seen_repos.add(fn)
                    out_f.write(json.dumps({
                        "full_name": fn, "created": it["created_at"],
                        "stars": it["stargazers_count"],
                        "license": (it.get("license") or {}).get("key"),
                        "size_kb": it.get("size", 0),
                        "clone_url": it["clone_url"], "_window": wkey}) + "\n")
                    got += 1
                if len(items) < 100:
                    break
                time.sleep(2.2)  # 30 req/min ceiling, stay under
            out_f.flush()
            n_new += got
            time.sleep(2.2)
    print(f"[{time.strftime('%H:%M:%S')}] {d}..{d2} done — total {len(seen_repos)} repos "
          f"(+{n_new} this run)", flush=True)
    d = d2 + timedelta(days=1)

print(f"discovery complete: {len(seen_repos)} repos -> {OUT}")
