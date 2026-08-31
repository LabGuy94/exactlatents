"""rt_grade set-grading for multi-name (I6) answers.

Run: uv run python -m pytest tests/test_rt_grade_i6.py -q
     (or plain: uv run python tests/test_rt_grade_i6.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.qa_filters import rt_grade


def _rec(answer, answer_type="name", intent="I6"):
    # Multi-name answers use set grading only for the I6 intent, so the helper
    # defaults to that branch.
    return {"answer": answer, "answer_type": answer_type, "intent": intent}


MULTI = _rec("get_num_params, get_trainable_parameters")


def test_exact_order():
    assert rt_grade("get_num_params, get_trainable_parameters", MULTI)


def test_reversed_order():
    assert rt_grade("get_trainable_parameters, get_num_params", MULTI)


def test_prose_embedded():
    assert rt_grade(
        "The functions get_trainable_parameters and get_num_params "
        "both match.", MULTI)


def test_missing_one_name_fails():
    assert not rt_grade("get_num_params", MULTI)
    assert not rt_grade("get_trainable_parameters and something_else", MULTI)


def test_empty_pred_fails():
    assert not rt_grade("", MULTI)


def test_overlapping_names():
    # "add" must NOT be credited by its occurrence inside "add_numbers"
    rec = _rec("add, add_numbers")
    assert not rt_grade("add_numbers", rec)
    assert not rt_grade("only add_numbers matches", rec)
    assert rt_grade("add and add_numbers", rec)
    assert rt_grade("add_numbers, add", rec)


def test_short_name_boundaries():
    rec = _rec("a, task_type")
    assert not rt_grade("task_type", rec)          # 'a' inside 'task' is no hit
    assert rt_grade("a, task_type", rec)
    assert rt_grade("both a and task_type", rec)


def test_extra_sibling_rejected():
    rec = _rec("foo, bar")
    rec["sibling_names"] = ["foo", "bar", "baz", "quux"]
    assert not rt_grade("foo, bar, baz", rec)      # hallucinated extra match
    assert rt_grade("foo and bar", rec)
    # non-candidate prose words never count as extra mentions
    assert rt_grade("the functions foo and bar both do this", rec)


def test_normalized_duplicate_fails():
    # "Add, add" collapses to one name under normalization: ungradeable
    assert not rt_grade("Add and add", _rec("Add, add"))


def test_single_name_unchanged():
    rec = _rec("get_num_params")
    assert rt_grade("get_num_params", rec)
    assert rt_grade("The answer is get_num_params.", rec)
    assert not rt_grade("get_trainable_parameters", rec)
    assert not rt_grade("", rec)


def test_substring_type_unchanged():
    # substring answers containing commas must keep whole-string grading
    rec = _rec("a, b, c", answer_type="substring")
    assert rt_grade("a, b, c", rec)
    assert not rt_grade("c, b, a", rec)


# --- Additional grading boundaries ----------------------------------------

def test_single_name_boundary():
    # A short name must not be credited inside a longer identifier.
    rec = _rec("add", intent="I1")
    assert not rt_grade("add_numbers", rec)
    assert rt_grade("add", rec)
    assert rt_grade("it is add.", rec)


def test_single_name_sibling_rejected():
    # A stacked single-name answer must not accept a sibling mention.
    rec = _rec("foo", intent="I1")
    rec["sibling_names"] = ["foo", "bar", "baz"]
    assert not rt_grade("foo and bar", rec)
    assert rt_grade("foo", rec)
    assert rt_grade("the function foo does this", rec)


def test_ordered_list_outside_i6():
    # Parameter lists outside I6 remain ordered.
    rec = _rec("a, b", intent="A3")
    assert rt_grade("a, b", rec)
    assert rt_grade("a,b", rec)
    assert not rt_grade("b, a", rec)
    assert not rt_grade("a, b, c", rec)


def test_blank_ground_truth_fails():
    # Blank ground truth must fail rather than match every prediction.
    assert not rt_grade("anything", _rec(" ", answer_type="substring",
                                         intent="A12"))
    assert not rt_grade("anything", _rec("\n", answer_type="substring",
                                         intent="H1"))


def test_symbolic_leak_detected():
    from dataset.qa_filters import leaks
    # Short symbolic answers still participate in leak detection.
    assert leaks("Does it return []?", "[]", "substring", [])
    assert leaks("What does the # character start?", "#", "substring", [])
    # "." inside identifiers must not false-positive
    assert not leaks("What does obj.attr hold?", ".", "substring", [])


def test_mcq_letter_within_options():
    from dataset.qa_filters import validate_record
    # An answer letter must index an existing option.
    base = {"row_idx": 0, "intent": "H3", "held_out": False,
            "question": "which style is used here?", "answer": "D",
            "answer_type": "letter", "context": {"kind": "solo"},
            "source": "mcq", "options": ["x", "y"]}
    assert any("beyond" in e for e in validate_record(base))
    base2 = dict(base, answer="B")
    assert validate_record(base2) == []


def test_validate_blank_answer_and_held_out():
    from dataset.qa_filters import validate_record
    base = {"row_idx": 0, "intent": "A12", "held_out": False,
            "question": "what is the docstring here?", "answer": " ",
            "answer_type": "substring", "context": {"kind": "solo"},
            "source": "ast"}
    assert any("blank" in e for e in validate_record(base))
    # The held-out flag must agree with the intent partition.
    ho = dict(base, answer="ok", intent="F2", held_out=False)
    assert any("held_out" in e for e in validate_record(ho))
    ho2 = dict(base, answer="ok", intent="F2", held_out=True)
    assert validate_record(ho2) == []


def test_collapsed_multiname_invalid():
    from dataset.qa_filters import validate_record
    # Duplicate names that collapse under normalization are ungradeable.
    base = {"row_idx": 0, "intent": "I6", "held_out": False,
            "question": "which functions do the described thing?",
            "answer": "Filter, filter", "answer_type": "name",
            "context": {"kind": "stack", "stack_row_idxs": [1, 2],
                        "target_row_idx": 1},
            "source": "ast"}
    assert any("collapses" in e for e in validate_record(base))


if __name__ == "__main__":
    for name, fn in sorted(
            (k, v) for k, v in globals().items() if k.startswith("test_")):
        fn()
        print(f"{name}: ok")
    print("all tests passed")
