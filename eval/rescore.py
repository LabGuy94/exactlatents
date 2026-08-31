#!/usr/bin/env python3
"""Recompute exactness counts from reconstruction JSONL receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compressor.exactness import byte_exact, code_exact  # noqa: E402

REFERENCE_FIELDS = ("code", "reference", "original", "target")
GENERATION_FIELDS = ("gen", "generation", "reconstruction", "pred", "prediction")
IDENTITY_FIELDS = ("func_name", "name", "id", "pair_id")


def first_field(row: dict, candidates: tuple[str, ...]) -> tuple[str | None, object | None]:
    for field in candidates:
        if field in row:
            return field, row[field]
    return None, None


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    return [row for row in rows if not any(key.startswith("_") for key in row)]


def row_identity(row: dict) -> object | None:
    return first_field(row, IDENTITY_FIELDS)[1]


def reference_value(row: dict) -> tuple[str | None, str | None]:
    field, value = first_field(row, REFERENCE_FIELDS)
    return field, value if isinstance(value, str) else None


def generation_value(row: dict) -> tuple[str | None, str | None]:
    field, value = first_field(row, GENERATION_FIELDS)
    return field, value if isinstance(value, str) else None


def dedupe_reference_rows(rows: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for row in rows:
        _, reference = reference_value(row)
        key = (row_identity(row), reference)
        if reference is not None and key in seen:
            continue
        if reference is not None:
            seen.add(key)
        result.append(row)
    return result


def attach_references(rows: list[dict], references: list[dict]) -> list[tuple[dict, str]]:
    references = dedupe_reference_rows(references)
    if len(rows) == len(references):
        pairs = []
        for index, (row, reference_row) in enumerate(zip(rows, references)):
            row_id = row_identity(row)
            reference_id = row_identity(reference_row)
            if row_id is not None and reference_id is not None and row_id != reference_id:
                raise ValueError(
                    f"row {index}: receipt identity {row_id!r} != reference identity {reference_id!r}"
                )
            _, reference = reference_value(reference_row)
            if reference is None:
                raise ValueError(f"row {index}: reference receipt has no source-code field")
            pairs.append((row, reference))
        return pairs

    by_identity: dict[object, str] = {}
    for reference_row in references:
        identity = row_identity(reference_row)
        _, reference = reference_value(reference_row)
        if identity is None or reference is None:
            continue
        if identity in by_identity and by_identity[identity] != reference:
            raise ValueError(f"ambiguous reference identity {identity!r}")
        by_identity[identity] = reference
    pairs = []
    for index, row in enumerate(rows):
        identity = row_identity(row)
        if identity not in by_identity:
            raise ValueError(f"row {index}: no reference for identity {identity!r}")
        pairs.append((row, by_identity[identity]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="Reconstruction JSONL receipt.")
    parser.add_argument(
        "--references",
        type=Path,
        help="Companion JSONL providing source code when the receipt only stores generations.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Count duplicate identity/source rows instead of keeping the first.",
    )
    args = parser.parse_args()

    rows = load_rows(args.receipt)
    if not rows:
        raise SystemExit(f"{args.receipt}: no receipt rows")
    if not args.keep_duplicates:
        inline_rows = [row for row in rows if reference_value(row)[1] is not None]
        if len(inline_rows) == len(rows):
            rows = dedupe_reference_rows(rows)

    if all(reference_value(row)[1] is not None for row in rows):
        pairs = [(row, reference_value(row)[1]) for row in rows]
        reference_source = str(args.receipt)
    elif args.references:
        pairs = attach_references(rows, load_rows(args.references))
        reference_source = str(args.references)
    else:
        raise SystemExit(
            "receipt has no source-code field; pass --references with a companion JSONL"
        )

    byte_count = 0
    code_count = 0
    reference_fields = set()
    generation_fields = set()
    for index, (row, reference) in enumerate(pairs):
        reference_field, inline_reference = reference_value(row)
        generation_field, generation = generation_value(row)
        if generation is None:
            raise SystemExit(f"row {index}: no generation field")
        if reference_field is not None and inline_reference is not None:
            reference_fields.add(reference_field)
        else:
            reference_fields.add("companion")
        generation_fields.add(generation_field)
        byte_count += byte_exact(reference, generation)
        code_count += code_exact(reference, generation)

    result = {
        "receipt": str(args.receipt),
        "references": reference_source,
        "rows": len(pairs),
        "byte_exact": byte_count,
        "code_exact": code_count,
        "reference_fields": sorted(reference_fields),
        "generation_fields": sorted(generation_fields),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
