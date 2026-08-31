#!/usr/bin/env python3
"""Build the frozen OOD-600 parameter-influence question bank.

Question extraction, template phrasing, and contract checks reuse the released
``dataset`` package. The builder adapts published OOD receipt rows to the
corpus-row shape expected by those modules and writes deterministic gzip JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "dataset"
sys.path.insert(0, str(DATASET))
sys.dont_write_bytecode = True

import qa_extract  # noqa: E402
import qa_filters  # noqa: E402
import qa_phrase  # noqa: E402

SEED = 20260830
MAX_PHRASE_ATTEMPTS = 64
INPUT = REPO / "receipts" / "ood600" / "ood600_pf4_gens.jsonl"
OUT_DIR = REPO / "receipts" / "ood_qa"
QUESTIONS = OUT_DIR / "questions_frozen.jsonl.gz"
INPUT_DISPLAY = "receipts/ood600/ood600_pf4_gens.jsonl"
EXTRACTOR_PATH = (
    "dataset.qa_extract.make_ctx -> dataset.qa_extract.extract_row -> "
    "dataset.qa_extract.REGISTRY['F2'].extract (_f2) -> "
    "dataset.qa_extract.fact_to_triple"
)
FILTER_PATH = [
    "dataset.qa_extract.leak_check (inside extract_row)",
    "dataset.qa_filters.repair",
    "dataset.qa_filters.format_reject",
    "dataset.qa_filters.leaks",
    "dataset.qa_filters.to_contract + validate_record",
]


def json_line(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def load_targets() -> tuple[list[tuple[int, dict]], int]:
    with INPUT.open(encoding="utf-8") as handle:
        raw = [(line_no, json.loads(line)) for line_no, line in enumerate(handle, 1)]
    if len(raw) != 601:
        raise RuntimeError(f"expected 601 raw rows, found {len(raw)}")
    first_key = (raw[0][1].get("func_name"), raw[0][1].get("code"))
    second_key = (raw[1][1].get("func_name"), raw[1][1].get("code"))
    if first_key != second_key:
        raise RuntimeError("the known duplicated first row is no longer rows 1-2")
    targets = [raw[0], *raw[2:]]
    if len(targets) != 600:
        raise RuntimeError(f"expected 600 targets after first-row dedupe, found {len(targets)}")
    return targets, len(raw)


def phrase_candidates(pair: dict) -> list[tuple[str, str, int]]:
    """Collect every actual F2 mock template through mock_question()."""
    expected = len(qa_phrase.INTENT_TASKS["F2"][1])
    seen: set[str] = set()
    candidates: list[tuple[str, str, int]] = []
    for offset in range(MAX_PHRASE_ATTEMPTS):
        phrase_seed = SEED + offset
        question, template_id = qa_phrase.mock_question(pair, phrase_seed)
        question = qa_filters.repair(question)
        if question in seen:
            continue
        seen.add(question)
        candidates.append((question, template_id, phrase_seed))
        if len(candidates) == expected:
            break
    if len(candidates) != expected:
        raise RuntimeError(
            f"found {len(candidates)} of {expected} F2 templates after "
            f"{MAX_PHRASE_ATTEMPTS} deterministic attempts"
        )
    return candidates


def filter_candidate(pair: dict, question: str) -> tuple[dict | None, str | None]:
    why = qa_filters.format_reject(question)
    if why:
        return None, f"format:{why}"
    if qa_filters.leaks(
        question, pair["answer"], pair["answer_type"], pair.get("extra_leak_terms", [])
    ):
        return None, "leak"
    raw = dict(pair, question=question)
    contract = qa_filters.to_contract(raw)
    errors = qa_filters.validate_record(contract)
    if errors:
        return None, f"contract:{errors[0]}"
    return contract, None


def skip_reason_for_inapplicable(ctx: qa_extract.FnCtx) -> str:
    user_params = [name for name in ctx.params() if name not in ("self", "cls")]
    if not user_params:
        return "no_params"
    if not ctx.returns_with_value():
        return "f2_not_applicable:no_value_return"
    return "extraction_failure:f2_emitted_nothing"


def build() -> tuple[str, dict]:
    targets, raw_count = load_targets()
    questions: list[dict] = []
    skipped: list[dict] = []
    skip_histogram: Counter[str] = Counter()
    filter_rejections: Counter[str] = Counter()
    questions_per_function: Counter[int] = Counter()
    gold_histogram: Counter[str] = Counter()
    covered_targets: set[int] = set()
    normalized_count = 0
    extracted_count = 0

    for target_idx, (source_row, target) in enumerate(targets):
        corpus_row = {
            "func_name": target["func_name"],
            "code": target["code"],
            "qualified": f"ood600.{target_idx:04d}.{target['func_name']}",
        }
        try:
            ctx = qa_extract.make_ctx(corpus_row, target_idx)
            facts, extraction_stats = qa_extract.extract_row(ctx)
        except Exception as exc:
            reason = f"extraction_failure:{type(exc).__name__}"
            skipped.append(
                {"source_row": source_row, "func_name": target["func_name"], "reason": reason}
            )
            skip_histogram[reason] += 1
            continue

        if ctx.normalized:
            normalized_count += 1
        f2_facts = [fact for fact in facts if fact.intent == "F2"]
        if not f2_facts:
            reason = skip_reason_for_inapplicable(ctx)
            if "F2" in extraction_stats["applicable"] and reason.startswith("f2_not_applicable"):
                reason = "extraction_failure:f2_emitted_nothing"
            skipped.append(
                {"source_row": source_row, "func_name": target["func_name"], "reason": reason}
            )
            skip_histogram[reason] += 1
            continue
        if len(f2_facts) != 1:
            raise RuntimeError(
                f"source row {source_row} emitted {len(f2_facts)} F2 facts; expected exactly one"
            )

        fact = f2_facts[0]
        extracted_count += 1
        triple = qa_extract.fact_to_triple(fact, corpus_row, target_idx)
        pair = qa_phrase.build_pair(triple, {target_idx: corpus_row})
        if pair is None:
            reason = "extraction_failure:pair_assembly"
            skipped.append(
                {"source_row": source_row, "func_name": target["func_name"], "reason": reason}
            )
            skip_histogram[reason] += 1
            continue

        accepted_for_target = 0
        target_filter_reasons: list[str] = []
        for question, template_id, phrase_seed in phrase_candidates(pair):
            contract, rejection = filter_candidate(pair, question)
            if rejection:
                filter_rejections[rejection] += 1
                target_filter_reasons.append(rejection)
                continue
            if contract is None:
                raise AssertionError("accepted candidate has no contract record")
            gold = pair["answer"]
            questions.append(
                {
                    "func_name": target["func_name"],
                    "code": target["code"],
                    "question": contract["question"],
                    "gold": gold,
                    "intent": "F2",
                    "answer_type": contract["answer_type"],
                    "grading": {
                        "method": "dataset.qa_filters.rt_grade",
                        "ordered_multi_name": "," in gold,
                        "source": contract["source"],
                    },
                    "provenance": {
                        "input": INPUT_DISPLAY,
                        "input_row": source_row,
                        "adapted_row_idx": target_idx,
                        "extractor": EXTRACTOR_PATH,
                        "phrasing": "dataset.qa_phrase.mock_question",
                        "filters": FILTER_PATH,
                    },
                    "notes": {
                        "base_seed": SEED,
                        "phrase_seed": phrase_seed,
                        "template_id": template_id,
                        "fact_pair_id": pair["pair_id"],
                        "held_out": bool(pair["held_out"]),
                        "normalized_for_parse": bool(ctx.normalized),
                    },
                }
            )
            accepted_for_target += 1
            gold_histogram["none" if gold == "none" else "named"] += 1

        if accepted_for_target:
            covered_targets.add(target_idx)
            questions_per_function[accepted_for_target] += 1
        else:
            suffix = ",".join(sorted(set(target_filter_reasons))) or "unknown"
            reason = f"filtered_out:{suffix}"
            skipped.append(
                {"source_row": source_row, "func_name": target["func_name"], "reason": reason}
            )
            skip_histogram[reason] += 1

    if len(covered_targets) + len(skipped) != len(targets):
        raise AssertionError("covered + skipped target accounting does not equal 600")
    if any(count < 1 or count > 3 for count in questions_per_function):
        raise AssertionError("question count per covered function fell outside 1-3")

    question_text = "".join(json_line(row) for row in questions)
    questions_sha256 = hashlib.sha256(question_text.encode("utf-8")).hexdigest()
    input_sha256 = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    stats = {
        "raw_count": raw_count,
        "target_count": len(targets),
        "covered": len(covered_targets),
        "questions": len(questions),
        "skipped": skipped,
        "skip_histogram": skip_histogram,
        "filter_rejections": filter_rejections,
        "questions_per_function": questions_per_function,
        "gold_histogram": gold_histogram,
        "normalized_count": normalized_count,
        "extracted_count": extracted_count,
        "input_sha256": input_sha256,
        "questions_sha256": questions_sha256,
    }
    return question_text, stats




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute in memory and verify the frozen artifacts byte-for-byte",
    )
    args = parser.parse_args()
    question_text, stats = build()

    if args.check:
        if not QUESTIONS.exists():
            raise SystemExit(f"missing frozen artifact: {QUESTIONS}")
        with gzip.open(QUESTIONS, "rt", encoding="utf-8") as handle:
            frozen_text = handle.read()
        if frozen_text != question_text:
            raise SystemExit(f"frozen artifact mismatch: {QUESTIONS}")
        print(
            f"PASS: byte-identical question rows; {stats['covered']}/600 functions, "
            f"{stats['questions']} questions"
        )
        return

    if QUESTIONS.exists():
        raise SystemExit(
            f"refusing to overwrite frozen artifact: {QUESTIONS}; use --check"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QUESTIONS.write_bytes(gzip.compress(question_text.encode("utf-8"), mtime=0))
    print(
        f"FROZEN: {stats['covered']}/600 functions, {stats['questions']} questions; "
        f"questions={QUESTIONS.relative_to(REPO)}"
    )


if __name__ == "__main__":
    main()
