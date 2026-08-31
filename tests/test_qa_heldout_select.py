"""Held-out QA record-selection tests.

Synthetic shard pairs exercise positional metadata joins, held-out
flag/intent consistency, round-trip partitions, contract exclusion,
trained-comparison sampling, and torn-shard detection.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "qa_eval", ROOT / "eval" / "qa.py"
)
qa_heldout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qa_heldout)


def _rec(intent, held, answer="foo", kind="solo", **kw):
    ctx = {"kind": kind}
    if kind == "stack":
        ctx.update(stack_row_idxs=[0, 1], target_row_idx=0)
    r = {"row_idx": 0, "intent": intent, "held_out": held, "question": "What name?",
         "answer": answer, "answer_type": "name", "context": ctx, "source": "ast"}
    r.update(kw)
    return r


def _write(dirp, shards):
    """shards: list of [(rec, rt_pass), ...] per shard index"""
    for i, rows in enumerate(shards):
        with open(dirp / f"qa_pairs_{i:04d}.jsonl", "w") as pf, \
             open(dirp / f"qa_pairs_meta_{i:04d}.jsonl", "w") as mf:
            for j, (r, rt) in enumerate(rows):
                pf.write(json.dumps(r) + "\n")
                mf.write(json.dumps({"pair_id": f"p{i}_{j}", "rt_pass": rt}) + "\n")


def test_partition_and_join(tmp_path):
    shards = [
        [(_rec("F2", True), True), (_rec("F2", True), False),
         (_rec("D7", True), True), (_rec("D7", True), None)],
        [(_rec("G4", False), True),            # trained pool
         (_rec("I1", False, kind="stack"), True),   # stack: not in comparison pool
         (_rec("G4", False), False),           # rt fail: not in pool
         (_rec("G4", False, answer="  "), True)],   # contract-bad: excluded
    ]
    _write(tmp_path, shards)
    held, adv, trained, stats = qa_heldout.select_records(tmp_path, trained_sample=10, seed=0)
    assert len(held) == 2 and all(r["rt_pass"] is True for r in held)
    assert len(adv) == 2                        # False AND None both advisory
    assert stats["held_out"] == 4 and stats["contract_excluded"] == 1
    assert stats["trained_pool"] == 1 and len(trained) == 1
    assert trained[0]["intent"] == "G4" and trained[0]["pair_id"] == "p1_0"
    assert stats["held_intents"] == {"F2": {"n": 2, "rt_true": 1},
                                     "D7": {"n": 2, "rt_true": 1}}


def test_flag_intent_mismatch_dies(tmp_path):
    _write(tmp_path, [[(_rec("F2", True), True), (_rec("F2", False), True)]])
    with pytest.raises(AssertionError, match="mismatches intent"):
        qa_heldout.select_records(tmp_path)


def test_torn_meta_shard_dies(tmp_path):
    _write(tmp_path, [[(_rec("F2", True), True)]])
    with open(tmp_path / "qa_pairs_meta_0000.jsonl", "a") as f:
        f.write(json.dumps({"pair_id": "extra", "rt_pass": True}) + "\n")
    with pytest.raises(AssertionError, match="positional join torn"):
        qa_heldout.select_records(tmp_path)


def test_stack_heldout_dies(tmp_path):
    _write(tmp_path, [[(_rec("F2", True), True),
                       (_rec("D7", True, kind="stack",
                             sibling_names=None), True)]])
    with pytest.raises(AssertionError, match="stack context"):
        qa_heldout.select_records(tmp_path)


def test_empty_denominator_dies(tmp_path):
    _write(tmp_path, [[(_rec("F2", True), False)]])
    with pytest.raises(AssertionError, match="empty held-out"):
        qa_heldout.select_records(tmp_path)


def test_seeded_draw_deterministic(tmp_path):
    rows = [[(_rec("F2", True), True)] +
            [(_rec("G4", False, answer=f"n{k}"), True) for k in range(50)]]
    _write(tmp_path, rows)
    a = qa_heldout.select_records(tmp_path, trained_sample=5, seed=1)[2]
    b = qa_heldout.select_records(tmp_path, trained_sample=5, seed=1)[2]
    assert [r["pair_id"] for r in a] == [r["pair_id"] for r in b]
