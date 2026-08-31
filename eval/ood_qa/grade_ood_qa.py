#!/usr/bin/env python3
"""Regrade the released OOD function-QA receipts with the local grader."""

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

EXPECTED_SHA256 = "fc887b7865d8a4c8794549322f149df84d799189b8d39c7c9706cb979fa79e2e"
EXPECTED_N = 1268
ARMS = ("vec", "ft_text", "stock_text")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict:
    both = sum(a and b for a, b in zip(left, right))
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(not a and b for a, b in zip(left, right))
    neither = len(left) - both - left_only - right_only
    discordant = left_only + right_only
    if discordant:
        low = min(left_only, right_only)
        p = min(1.0, 2.0 * sum(math.comb(discordant, k) for k in range(low + 1)) / (2 ** discordant))
    else:
        p = 1.0
    return {
        "both_right": both,
        "left_only": left_only,
        "right_only": right_only,
        "both_wrong": neither,
        "discordant": discordant,
        "exact_two_sided_p": p,
    }




def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=root / "receipts" / "ood_qa" / "questions_frozen.jsonl.gz",
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=root / "receipts" / "ood_qa",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "results" / "ood_qa_grade.jsonl",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(root))
    from dataset.qa_filters import rt_grade

    question_sha = sha256_file(args.questions)
    assert question_sha == EXPECTED_SHA256, f"frozen questions changed: {question_sha}"
    opener = gzip.open if args.questions.suffix == ".gz" else open
    with opener(args.questions, "rt", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    assert len(questions) == EXPECTED_N

    stats = {}
    for arm in ARMS:
        path = args.receipts_dir / f"{arm}.jsonl"
        with path.open(encoding="utf-8") as f:
            receipts = [json.loads(line) for line in f if line.strip()]
        assert len(receipts) == EXPECTED_N, f"{arm}: {len(receipts)} rows"
        raw_flags, corrected_flags = [], []
        for i, (receipt, question) in enumerate(zip(receipts, questions)):
            assert receipt["index"] == i and receipt["arm"] == arm
            assert (receipt["func_name"], receipt["question"], receipt["gold"]) == (
                question["func_name"], question["question"], question["gold"]
            ), f"{arm}:{i}: frozen order mismatch"
            grader_input = dict(question)
            grader_input["answer"] = question["gold"]
            raw_ok = bool(rt_grade(receipt["first_line_pred"], grader_input))
            corrected_ok = bool(rt_grade(receipt["corrected_pred"], grader_input))
            assert raw_ok == receipt["ok_first_line"], f"{arm}:{i}: stored/local raw grade mismatch"
            assert corrected_ok == receipt["ok_corrected"], f"{arm}:{i}: stored/local corrected grade mismatch"
            raw_flags.append(raw_ok)
            corrected_flags.append(corrected_ok)
        stats[arm] = {
            "raw_flags": raw_flags,
            "corrected_flags": corrected_flags,
            "raw_ok": sum(raw_flags),
            "corrected_ok": sum(corrected_flags),
        }

    raw_mc = exact_mcnemar(stats["vec"]["raw_flags"], stats["ft_text"]["raw_flags"])
    corrected_mc = exact_mcnemar(stats["vec"]["corrected_flags"], stats["ft_text"]["corrected_flags"])
    result = {
        "question_sha256": question_sha,
        "stats": {
            arm: {
                "n": EXPECTED_N,
                "raw_ok": stats[arm]["raw_ok"],
                "corrected_ok": stats[arm]["corrected_ok"],
            }
            for arm in ARMS
        },
        "mcnemar_raw": raw_mc,
        "mcnemar_corrected": corrected_mc,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(result, out=str(args.out)), sort_keys=True))


if __name__ == "__main__":
    main()
