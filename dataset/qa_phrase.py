"""QA phrasing pipeline — driver.

Converts extracted answer triples from ``dataset/qa_extract.py`` into
natural-language questions through an OpenAI-compatible endpoint. The model
writes questions only; answers remain the deterministic upstream values.

Stages:

  sample   quota-sample intent/function pairs from finalized triples, attach
           corpus code, and write sampled_pairs.jsonl plus sample_meta.json.
  phrase   consume sampled_pairs.jsonl, call the endpoint (or --mock), and
           append qa_raw.jsonl; phrase_done.ids makes this resumable.
  all      run sample followed by phrase.
  render-templates
           write the prompt and template bank to
           out/qa/PHRASING_TEMPLATES.md.

Output contract (qa_filters.py assembles the final qa_pairs_*.jsonl):
  {"row_idx": int, "intent": str, "held_out": bool, "question": str,
   "answer": str, "answer_type": "substring|number|yesno|name|letter|line_ref",
   "context": {"kind": "solo"|"stack", "stack_row_idxs": [...] iff stack,
               "target_row_idx": int iff stack},
   "source": "ast"|"exec"|"luna"|"mcq", "options": [...] iff MCQ}

Design notes:
- needs_luna triples (answer null: C1, C3, C2-not-runnable) are SKIPPED here;
  a separate conceptual-answer pipeline produces finalized
  ``triples_luna.jsonl`` records with answers filled and ``needs_luna=false``.
  Those records are consumed like every other source.
- directory mode consumes exactly: triples_ast_NNNN.jsonl shards +
  triples_exec.jsonl + triples_luna.jsonl + triples_i6.jsonl (finalized files
  only — raw and *_usage intermediate files are never eligible). A missing
  finalized file warns loudly and its group quota is redistributed.
- answer_type uses the fixed trainer contract: extractor type "list" maps to
  "name" for identifier-list intents or "substring" for code-list intents;
  "text" maps to "substring". Grading remains exact or normalized string match.
- Sampling, style selection, and mock generation are seeded by
  ``(--seed, pair_id)`` for byte-identical reruns over identical shards.

Usage:
  uv run python dataset/qa_phrase.py sample --triples data/corpus_v2/qa_full \
      --n 400000 --out-dir data/corpus_v2/qa_phrased
  uv run python dataset/qa_phrase.py phrase --out-dir data/corpus_v2/qa_phrased \
      --mock --plant-leaks 50
  uv run python dataset/qa_phrase.py render-templates
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Catalog constants
# --------------------------------------------------------------------------

GROUP_QUOTAS = {
    "A": 0.20, "B": 0.15, "C": 0.15, "D": 0.15,
    "E/F": 0.12, "G": 0.05, "H": 0.08, "I": 0.10,
}

HELD_OUT_INTENTS = {"F2", "D7"}  # phrased, marked held_out, never trained

# Answers too trivial for a containment leak check to be meaningful
# (mirrors qa_extract._TRIVIAL_ANSWERS philosophy; numbers stay checked as
# standalone tokens — the coverage report names the phrasing-stage filter as
# the second gate for small-integer answers).
TRIVIAL_LEAK_EXEMPT = frozenset({"none", "then", "else", "true", "false"})

# extractor answer_type -> pinned contract answer_type
_LIST_AS_NAME = {"A3", "A8", "C6", "F2", "F4"}


def map_answer_type(atype: str, intent: str) -> str:
    if atype == "list":
        return "name" if intent in _LIST_AS_NAME else "substring"
    if atype == "text":
        return "substring"
    return atype  # substring | number | yesno | name | letter | line_ref


def infer_source(intent: str, seed: dict) -> str:
    if "options" in seed:
        return "mcq"
    if intent.startswith("D") or (intent == "C2" and "other_args" in seed):
        return "exec"
    return "ast"


def pair_id_of(intent: str, row_idx: int, seed: dict) -> str:
    key = f"{intent}|{row_idx}|{json.dumps(seed, sort_keys=True)}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Style bank — ~20 surface directives per intent family
# --------------------------------------------------------------------------

BASE_STYLES = [
    ("terse", "Terse developer shorthand: under ten words, lowercase is fine, "
              "no pleasantries."),
    ("full", "One complete, grammatically polished interrogative sentence."),
    ("imperative", "An imperative instruction rather than a question (e.g. "
                   "'Name ...', 'State ...', 'Quote ...'). No question mark."),
    ("first-person", "First-person frame: begin with something like \"I'm "
                     "reading this code and ...\"."),
    ("review", "A code-review comment asking the author to clarify this fact."),
    ("newcomer", "A question from someone new to the codebase, slightly "
                 "unsure of the local terminology."),
    ("quiz", "A neutral, precise quiz item, exam-style."),
    ("chat", "A casual chat message to a teammate; contractions welcome."),
    ("docs-writer", "A technical writer double-checking a detail for "
                    "documentation."),
    ("debug", "Someone mid-debugging who needs this fact fast; a little "
              "urgency is fine."),
    ("interview", "An interviewer probing whether the candidate actually "
                  "read the code."),
    ("polite", "A polite request: 'Could you tell me ...' / 'Would you "
               "mind ...'."),
    ("noun-phrase", "A bare noun phrase plus question mark (e.g. 'Number of "
                    "parameters?')."),
    ("ticket", "Phrased like an issue-tracker title ending in a question "
               "mark."),
    ("confirm", "Seeking confirmation of a hunch WITHOUT stating the "
                "expected value (never assert the answer itself)."),
    ("spoken", "As you'd say it out loud in a standup: conversational "
               "rhythm, no written-prose stiffness."),
]

FAMILY_EXTRA_STYLES = {
    "identity": [
        ("caller-view", "Ask from the caller's perspective: what would "
                        "someone need to know to invoke this correctly?"),
        ("api-doc", "Framed as filling in a field of an API reference entry."),
        ("skim", "Framed as someone skimming the file who stopped at this "
                 "definition."),
        ("refactor", "Framed by someone about to refactor who wants this "
                     "fact pinned down first."),
    ],
    "span": [
        ("exact-quote", "Emphasize verbatim fidelity: the reply must quote "
                        "the source exactly, character for character."),
        ("cite", "Framed as needing the exact text to cite in a commit "
                 "message."),
        ("diff", "Framed as reconstructing a diff hunk and needing the "
                 "original text."),
        ("pin", "Framed as pinning down the precise source text before "
                "editing it."),
    ],
    "behavior": [
        ("what-if", "A 'what happens if/when ...' framing."),
        ("contract", "Framed as checking the function's contract rather "
                     "than its text."),
        ("edge", "Framed around edge-case curiosity."),
        ("caller-code", "From the perspective of code that is about to call "
                        "this function."),
    ],
    "mcq": [
        ("choose", "A stem that sets up choosing among options shown "
                   "separately (the options are NOT part of the stem)."),
        ("best-match", "Ask which of the given choices best matches; never "
                       "reveal any option text in the stem."),
        ("select-one", "Instruct the reader to select the single correct "
                       "choice from the options that follow."),
        ("identify", "Ask to identify the correct characterization among "
                     "the given alternatives."),
    ],
    "exec": [
        ("trace", "Framed as mentally tracing an actual run of the code."),
        ("predict", "Framed as predicting the outcome before running it."),
        ("repl", "Framed as sitting at a REPL about to evaluate the call."),
        ("unit-test", "Framed as writing the assertion for a unit test."),
    ],
    "flow": [
        ("path", "Framed around control-flow paths through the function."),
        ("dataflow", "Framed around how data moves through the function."),
        ("guard", "Framed around guard clauses and early exits."),
        ("static-analysis", "Framed as verifying a static-analysis finding "
                            "by eye."),
    ],
    "quality": [
        ("lint", "Framed as double-checking a linter warning by hand."),
        ("smell", "Framed as sniffing for a well-known Python code smell."),
        ("audit", "Framed as one item of a quick code-hygiene audit."),
        ("gotcha", "Framed around classic Python gotchas."),
    ],
    "docs": [
        ("doc-sync", "Framed as checking documentation against the code's "
                     "reality."),
        ("verbatim-doc", "Emphasize reproducing the documentation text "
                         "exactly as written."),
        ("style-guide", "Framed as a docs style-guide compliance check."),
        ("onboarding", "Framed as writing onboarding notes for a new hire."),
    ],
    "retrieval": [
        ("lookup", "Framed as hunting through this file for the right "
                   "helper."),
        ("delegate", "Framed as deciding which function to call for a task "
                     "at hand."),
        ("describe-first", "Lead with the behavioral description, then ask "
                           "which function it is."),
        ("verify-claim", "Framed as verifying a teammate's claim about "
                         "which function does what."),
    ],
}

FAMILY_STYLES = {
    fam: BASE_STYLES + extras for fam, extras in FAMILY_EXTRA_STYLES.items()
}

INTENT_FAMILY = {}
for _iid in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
             "A11", "A12"]:
    INTENT_FAMILY[_iid] = "identity"
for _iid in ["B1", "B2", "B3", "B4", "B5", "B6"]:
    INTENT_FAMILY[_iid] = "span"
for _iid in ["C1", "C2", "C3", "C4", "C6"]:
    INTENT_FAMILY[_iid] = "behavior"
for _iid in ["C5", "C7"]:
    INTENT_FAMILY[_iid] = "mcq"
for _iid in ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]:
    INTENT_FAMILY[_iid] = "exec"
for _iid in ["E1", "E2", "E3", "F1", "F2", "F3", "F4"]:
    INTENT_FAMILY[_iid] = "flow"
for _iid in ["G1", "G2", "G3", "G4"]:
    INTENT_FAMILY[_iid] = "quality"
for _iid in ["H1", "H2", "H3"]:
    INTENT_FAMILY[_iid] = "docs"
for _iid in ["I1", "I2", "I3", "I4", "I5", "I6"]:
    INTENT_FAMILY[_iid] = "retrieval"


# --------------------------------------------------------------------------
# Intent task cards — what the model is told to ask about, per intent.
# `seed` is the extractor's question_seed; task(seed) -> instruction text.
# Mock templates are .format()-ed with the seed (pipeline testing only).
# --------------------------------------------------------------------------

def _e1_kind(seed):
    return {"return None": "returns None early",
            "raise": "raises early",
            "exit loop": "exits the loop early"}.get(seed.get("exit_kind"),
                                                     "exits early")


def _c4_phrase(seed):
    return {"mutates an argument": "mutates any of its arguments",
            "recursive": "is recursive (calls itself)",
            "can return None": "can return None"}.get(
        seed.get("property"), seed.get("property", "has the property"))


INTENT_TASKS = {
    # -- group A: identity & structure ------------------------------------
    "A1": (lambda s: "Ask what this function is named (what it is called). "
                     "The expected answer is the function's name; the "
                     "question itself must not contain that name.",
           ["What is this function's name?",
            "Name of this function?",
            "State the name this function is defined under.",
            "What identifier is this function bound to?"]),
    "A2": (lambda s: "Ask how many parameters the function's signature "
                     "declares.",
           ["How many parameters does this function take?",
            "Parameter count?",
            "Count the parameters in this function's signature.",
            "How many arguments appear in the def line?"]),
    "A3": (lambda s: "Ask for the function's parameter names, in signature "
                     "order.",
           ["What are the parameter names, in order?",
            "List this function's parameters in order.",
            "Which names appear in the parameter list, in order?"]),
    "A4": (lambda s: f"Ask what default value the parameter "
                     f"'{s.get('param')}' has.",
           ["What default does parameter {param} get?",
            "Default value of {param}?",
            "State the default assigned to the {param} parameter."]),
    "A5": (lambda s: f"Ask whether the signature includes "
                     f"{s.get('feature')}.",
           ["Does the signature use {feature}?",
            "Is {feature} part of this function's signature?",
            "Any {feature} here?"]),
    "A6": (lambda s: "Ask what the declared return type annotation is.",
           ["What is the declared return annotation?",
            "Return type annotation?",
            "State the annotation after the arrow in the signature."]),
    "A7": (lambda s: f"Ask what type annotation the parameter "
                     f"'{s.get('param')}' carries.",
           ["What annotation does parameter {param} carry?",
            "Type annotation on {param}?",
            "State the declared type of the {param} parameter."]),
    "A8": (lambda s: "Ask which decorators are applied to the function.",
           ["Which decorators are applied to this function?",
            "Decorator list?",
            "Name the decorators stacked on this definition."]),
    "A9": (lambda s: "Ask whether the function is asynchronous (defined "
                     "with async def).",
           ["Is this function async?",
            "Defined with async def, yes or no?",
            "Is this a coroutine function?"]),
    "A10": (lambda s: ("Ask whether this function is a method of a class."
                       if s.get("aspect") == "is_method" else
                       "Ask which class this method belongs to."),
            ["Is this function a method of a class?",
             "Which class does this method belong to?",
             "Method or free function?"]),
    "A11": (lambda s: (f"Ask how many times the function calls "
                       f"{s.get('callee')}()."
                       if s.get("count_of") == "calls" else
                       f"Ask how many {s.get('count_of')} the function body "
                       f"contains."),
            ["How many {count_of} does the body contain?",
             "Count of {count_of} in this function?",
             "Tally the {count_of} here."]),
    "A12": (lambda s: f"Ask for the {_ordinal(s.get('n', 1))} string "
                      f"literal in source order (excluding the docstring), "
                      f"exactly as written.",
            ["What is string literal number {n} (excluding the docstring)?",
             "Quote the {nth} string literal exactly.",
             "Which string constant comes {nth} in source order?"]),
    # -- group B: span grounding ------------------------------------------
    "B1": (lambda s: f"Ask to quote line {s.get('line')} of the function "
                     f"exactly (1-based, counting from the def line).",
           ["Quote line {line} exactly.",
            "What does line {line} say, verbatim?",
            "Reproduce line {line} character for character."]),
    "B2": (lambda s: f"Ask to quote, verbatim, the condition of the "
                     f"{_ordinal(s.get('k', 1))} {s.get('construct')} "
                     f"statement.",
           ["Quote the condition of {construct} statement number {k}.",
            "What is the exact test of the {kth} {construct}?",
            "Verbatim condition of the {kth} {construct}?"]),
    "B3": (lambda s: f"Ask to quote the {_ordinal(s.get('k', 1))} return "
                     f"expression verbatim.",
           ["Quote return expression number {k} exactly.",
            "What does the {kth} return statement return, verbatim?",
            "Exact text of the {kth} return expression?"]),
    "B4": (lambda s: "Ask for the first line of the docstring, verbatim.",
           ["What is the docstring's first line, verbatim?",
            "Quote the opening line of the docstring.",
            "First docstring line, exactly as written?"]),
    "B5": (lambda s: f"Ask which lines the {_ordinal(s.get('k', 1))} "
                     f"{s.get('construct')} spans; the expected answer "
                     f"format is 'L<start>-L<end>' with 1-based line "
                     f"numbers counted from the def line.",
           ["Which lines does {construct} number {k} span (as L<a>-L<b>)?",
            "Line span of the {kth} {construct}?",
            "Give the L<start>-L<end> range of the {kth} {construct}."]),
    "B6": (lambda s: f"Ask whether the function body contains a "
                     f"{s.get('construct')}. (It does not; the expected "
                     f"answer is 'no' — phrase neutrally, do not hint.)",
           ["Does the body contain a {construct}?",
            "Any {construct} in this function?",
            "Is there a {construct} anywhere in here?"]),
    # -- group C: behavior & semantics ------------------------------------
    "C1": (lambda s: "Ask what this function computes / returns, described "
                     "conceptually (its overall behavior and result, not a "
                     "line-by-line walkthrough).",
           ["What does this function do, in a sentence or two?",
            "Describe what this function computes and returns.",
            "Summarize this function's behavior.",
            "What result does calling this function produce, conceptually?"]),
    "C3": (lambda s: "Ask what this function's purpose is — why it exists / "
                     "what role it plays (intent, not mechanics).",
           ["What is this function's purpose?",
            "Why would a codebase include this function?",
            "What problem is this function meant to solve?",
            "What is the point of this function?"]),
    "C2": (lambda s: (f"Ask what the function does / returns when the "
                      f"argument '{s.get('param')}' is empty"
                      + (f", with the call otherwise made as "
                         f"{s.get('other_args')}" if s.get("other_args")
                         else "") + "."),
           ["What happens when {param} is empty?",
            "Behavior for an empty {param}?",
            "What comes back if {param} has nothing in it?"]),
    "C4": (lambda s: f"Ask whether the function {_c4_phrase(s)}.",
           ["Does this function {property}?",
            "Would you say it {property}?",
            "True or false style: it {property}?"]),
    "C5": (lambda s: "Multiple-choice stem: ask which of the provided "
                     "descriptions best matches what this function does. "
                     "The options are shown separately — do NOT enumerate "
                     "or hint at any option text in the stem.",
           ["Which of the given descriptions matches this function?",
            "Pick the description that fits this code.",
            "Which listed summary is the right one for this function?"]),
    "C6": (lambda s: "Ask which exception types the function explicitly "
                     "raises.",
           ["Which exception types does it explicitly raise?",
            "What does this function raise?",
            "Name the exceptions raised directly in this body."]),
    "C7": (lambda s: f"Multiple-choice stem: ask what role the variable "
                     f"'{s.get('var')}' plays in this function. Options "
                     f"are shown separately — do not list them.",
           ["What role does the variable {var} play here?",
            "Which of the given roles fits {var}?",
            "How would you classify {var}'s job in this code?"]),
    # -- group D: execution -----------------------------------------------
    "D1": (lambda s: f"Ask what the function returns for the call "
                     f"{s.get('input')}.",
           ["What does {input} return?",
            "Result of calling {input}?",
            "Evaluate {input} — what comes back?"]),
    "D2": (lambda s: f"Ask for an input (a concrete call) that makes the "
                     f"function produce {s.get('output')}. The output "
                     f"value appears in the question by design; the "
                     f"answer is the input.",
           ["What input makes this function produce {output}?",
            "Give a call that yields {output}.",
            "Which arguments would return {output}?"]),
    "D3": (lambda s: f"Ask for the value of the variable '{s.get('var')}' "
                     f"immediately after line {s.get('after_line')} first "
                     f"executes, for the call {s.get('input')}.",
           ["For {input}, what is {var} right after line {after_line} "
            "first runs?",
            "Value of {var} after line {after_line} on the call {input}?",
            "Trace {input}: what does {var} hold once line {after_line} "
            "has executed the first time?"]),
    "D4": (lambda s: f"Ask whether line {s.get('line')} executes for the "
                     f"call {s.get('input')}.",
           ["Does line {line} run for {input}?",
            "For {input}, is line {line} reached?",
            "Would {input} ever hit line {line}?"]),
    "D5": (lambda s: f"Ask which branch — 'then' or 'else' — of the if "
                     f"statement on line {s.get('if_line')} runs for the "
                     f"call {s.get('input')}. The expected answer is the "
                     f"word 'then' or 'else'.",
           ["For {input}, does the if on line {if_line} take then or else?",
            "Which arm of line {if_line}'s if fires on {input}?",
            "then or else for the branch at line {if_line}, given "
            "{input}?"]),
    "D6": (lambda s: f"Ask how many times the body of the loop at line "
                     f"{s.get('loop_line')} runs for the call "
                     f"{s.get('input')}.",
           ["How many iterations does the loop at line {loop_line} do "
            "for {input}?",
            "For {input}, how often does line {loop_line}'s loop body "
            "run?",
            "Loop at line {loop_line}: iteration count on {input}?"]),
    "D7": (lambda s: f"Ask which exception type the call {s.get('input')} "
                     f"raises.",
           ["What exception does {input} raise?",
            "Which error type comes out of {input}?",
            "Calling {input} blows up with what exception?"]),
    # -- group E/F: control & data flow -----------------------------------
    "E1": (lambda s: f"Ask under what condition the function "
                     f"{_e1_kind(s)}; the expected answer is the guard "
                     f"condition, verbatim.",
           ["Under what condition does it {exit_kind} early?",
            "What guard triggers the early {exit_kind}?",
            "Quote the condition that causes the early {exit_kind}."]),
    "E2": (lambda s: "Ask whether the function contains unreachable code.",
           ["Is there unreachable code in this function?",
            "Any dead code here?",
            "Does any statement in this body never execute?"]),
    "E3": (lambda s: "Ask for the distinct return expressions, verbatim.",
           ["What distinct expressions can this function return?",
            "List the different return expressions, verbatim.",
            "Which exact expressions appear after 'return' here?"]),
    "F1": (lambda s: f"Ask on which line the variable '{s.get('var')}' is "
                     f"first assigned; expected answer format 'L<n>' with "
                     f"1-based lines counted from the def line.",
           ["On which line is {var} first assigned (answer as L<n>)?",
            "Where does {var} get assigned? Give L<n>.",
            "Line number (L<n>) of {var}'s first assignment?"]),
    "F2": (lambda s: "Ask which parameters influence the value the "
                     "function returns.",
           ["Which parameters influence the return value?",
            "Which inputs actually affect what comes back?",
            "Name the parameters the return value depends on."]),
    "F3": (lambda s: "Ask whether the function has unused variables or "
                     "parameters.",
           ["Are there unused variables or parameters here?",
            "Anything defined but never used?",
            "Does this function declare something it never reads?"]),
    "F4": (lambda s: "Ask which of its arguments the function mutates.",
           ["Which arguments does this function mutate?",
            "Does it modify any of its inputs — which?",
            "Name the parameters whose objects get changed in place."]),
    # -- group G: quality -------------------------------------------------
    "G1": (lambda s: "Ask whether any parameter has a mutable default "
                     "value.",
           ["Any mutable default parameter here?",
            "Does a parameter default to a mutable object?",
            "Is the mutable-default-argument pitfall present?"]),
    "G2": (lambda s: "Ask whether the function contains a bare except "
                     "clause.",
           ["Is there a bare except in this function?",
            "Any except clause without an exception type?",
            "Does this code swallow everything with a bare except?"]),
    "G3": (lambda s: ("Ask which Python builtin name the function shadows."
                      if s.get("aspect") == "which" else
                      "Ask whether the function shadows a Python builtin "
                      "name."),
           ["Does this function shadow a Python builtin?",
            "Which builtin name gets shadowed here?",
            "Any variable reusing a builtin's name?"]),
    "G4": (lambda s: "Ask whether the first parameter of this method is "
                     "something other than the conventional 'self'.",
           ["Is the first parameter of this method not self?",
            "Does this method break the self convention?",
            "First arg something other than self here?"]),
    # -- group H: docs ----------------------------------------------------
    "H1": (lambda s: "Ask to reproduce the function's docstring verbatim.",
           ["Reproduce the docstring exactly.",
            "What does the docstring say, word for word?",
            "Quote this function's docstring in full."]),
    "H2": (lambda s: "Ask whether the docstring's Raises section matches "
                     "the exceptions the code actually raises.",
           ["Does the documented Raises section match the actual raises?",
            "Is the Raises documentation accurate here?",
            "Do docstring and code agree on what gets raised?"]),
    "H3": (lambda s: ("Multiple-choice stem: ask which documentation style "
                      "the docstring follows. Options are shown separately "
                      "— do not name any style in the stem."
                      if s.get("aspect") == "style" else
                      "Ask whether the function has a docstring."),
           ["Does this function have a docstring?",
            "Which of the given styles does the docstring follow?",
            "Is there a docstring on this function?"]),
    # -- group I: descriptive retrieval over stacks ------------------------
    "I1": (lambda s: "You are shown several functions plus the TARGET "
                     "function's code. Write a question of the form "
                     "'which function <does X>?' where <does X> is YOUR "
                     "one-clause description of the target's behavior. "
                     "Describe behavior only — never use the target's "
                     "name or any part of it.",
           ["Which of these functions handles the described task "
            "(behavior: {mock_desc})?",
            "One of these functions {mock_desc} — which one?",
            "Find the function here that {mock_desc}."]),
    "I2": (lambda s: "You are shown a DECOY function's code (it is NOT "
                     "among the stack's functions). Write a question "
                     "asking whether any function in the given set does "
                     "what the decoy does, phrased via YOUR one-clause "
                     "description of the decoy's behavior. Never use the "
                     "decoy's name. The honest answer is 'no' — phrase "
                     "neutrally, do not hint.",
           ["Does any function here {mock_desc}?",
            "Is there a function in this set that {mock_desc}?",
            "Do any of these {mock_desc}?"]),
    "I3": (lambda s: "Multiple-choice stem over a set of functions: "
                     "describe the target function's behavior in one "
                     "clause (never its name) and ask which of the listed "
                     "function names it is. Options are shown separately.",
           ["Which listed function {mock_desc}?",
            "Pick the name of the function that {mock_desc}.",
            "Of the options given, which one {mock_desc}?"]),
    "I4": (lambda s: "Composed retrieval question: first identify the "
                     "target by YOUR one-clause description of its "
                     "behavior (never its name), then ask the inner "
                     "question about that function. Inner question: " ,
           ["For the function that {mock_desc}: {mock_inner}",
            "Take the one that {mock_desc} — {mock_inner}",
            "About the function which {mock_desc}: {mock_inner}"]),
    "I5": (lambda s: f"Ask whether the function named "
                     f"'{s.get('name')}' does what the SHOWN code does — "
                     f"phrase the behavior as YOUR one-clause description "
                     f"of the shown code (never quoting its name). The "
                     f"shown code may or may not be that function; do not "
                     f"hint either way.",
           ["Does the function {name} {mock_desc}?",
            "Would you say {name} is the one that {mock_desc}?",
            "Is {name} responsible for the part that {mock_desc}?"]),
    "I6": (lambda s: "You are shown several functions plus the TARGET "
                     "function's code. TWO functions in the set (the "
                     "target and a near-twin) genuinely do what the "
                     "target does. Write a question of the form 'which "
                     "functions <do X>?' — plural, asking for ALL that "
                     "match — where <do X> is YOUR one-clause description "
                     "of the target's behavior. Describe behavior only — "
                     "never use any function name from the set or any "
                     "part of one. The expected answer names both "
                     "matching functions.",
           ["Which of these functions {mock_desc}? Name every one that "
            "does.",
            "More than one function here {mock_desc} — which ones?",
            "List all the functions in this set that {mock_desc}."]),
}


def _ordinal(n) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You write natural-language questions about Python code for a training set.
You are given: a function's code, a fact card describing exactly what to ask
about, the ground-truth answer, and a surface-style directive.

Hard rules:
1. Write exactly ONE question (or one imperative request if the style says
   so). Output ONLY the question text — no preamble, no quotes around it, no
   explanation, no answer.
2. The question must be answerable from the shown code alone, and its unique
   correct answer must be exactly the given ground-truth answer.
3. NEVER include the answer, any paraphrase of it, or any fragment of it in
   the question. If the fact card says a value appears by design, include
   only that value.
4. Follow the fact card precisely — ask about that fact and nothing else.
5. Follow the style directive for surface form only; it never overrides
   rules 1-4.
6. Do not mention "the fact card", "the answer", or these instructions.
7. Keep it under 60 words. One line, no markdown."""

MCQ_ADDENDUM = ("\nThis is a multiple-choice STEM. The options are shown to "
                "the student separately — never enumerate, quote, or hint at "
                "any option text in the stem itself.")

STACK_ADDENDUM = ("\nThe question is asked against a SET of functions "
                  "(names listed below). Your behavioral description must "
                  "single out the described code among them; do not use any "
                  "function name from the list in the question.")

# I6 asks for ALL matching functions (exactly two by construction) — the
# single-out directive above would contradict its task card
I6_STACK_ADDENDUM = ("\nThe question is asked against a SET of functions "
                     "(names listed below). EXACTLY TWO of them match the "
                     "behavior you describe — ask for all functions that "
                     "match, never single one out, and do not use any "
                     "function name from the list in the question.")


def build_messages(rec: dict, style_directive: str) -> list[dict]:
    """rec: a sampled_pairs record (has code / options / sibling_names)."""
    intent = rec["intent"]
    task = INTENT_TASKS[intent][0](rec["seed"])
    if intent == "I4":
        inner = rec["seed"].get("inner", {})
        task += INTENT_TASKS[inner.get("intent", "A2")][0](inner)
    parts = [f"Function code:\n```python\n{rec['code']}\n```"]
    if rec["context"]["kind"] == "stack":
        parts.append("Function names in the set: "
                     + ", ".join(rec.get("sibling_names", [])))
    parts.append(f"Fact card: {task}")
    if rec.get("options"):
        parts.append("Options (context only; never reveal in the stem): "
                     + " | ".join(rec["options"]))
    parts.append(f"Ground-truth answer (do NOT use it in the question): "
                 f"{rec['answer']}")
    style = style_directive
    extra = ""
    if rec.get("options"):
        extra += MCQ_ADDENDUM
    if rec["context"]["kind"] == "stack":
        extra += I6_STACK_ADDENDUM if intent == "I6" else STACK_ADDENDUM
    parts.append(f"Style directive: {style}{extra}")
    parts.append("Write the question now.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)}]


def choose_style(pair_id: str, intent: str, seed: int) -> tuple[str, str]:
    fam = INTENT_FAMILY[intent]
    styles = FAMILY_STYLES[fam]
    rng = random.Random(f"{seed}|style|{pair_id}")
    return styles[rng.randrange(len(styles))]


# --------------------------------------------------------------------------
# Quota sampler
# --------------------------------------------------------------------------

_RX_ROW = re.compile(r'"row_idx": (\d+)')
_RX_INTENT = re.compile(r'"intent": "([A-Z]\d+)"')
_RX_LUNA = re.compile(r'"needs_luna": true')


def triples_files(triples: str) -> list[Path]:
    """Enumerate finalized AST shards and exact finalized auxiliary files.

    Directory mode admits only ``triples_ast_NNNN.jsonl``,
    ``triples_exec.jsonl``, ``triples_luna.jsonl``, and
    ``triples_i6.jsonl``. Raw and usage intermediates are never eligible.
    """
    p = Path(triples)
    if p.is_dir():
        # fullmatch, not glob: triples_ast_[0-9]* would also admit e.g.
        # triples_ast_0_raw.jsonl
        files = sorted(f for f in p.glob("triples_ast_*.jsonl")
                       if re.fullmatch(r"triples_ast_\d+\.jsonl", f.name))
        for name in ("triples_exec.jsonl", "triples_luna.jsonl",
                     "triples_i6.jsonl"):
            if (p / name).exists():
                files.append(p / name)
            else:
                print(f"[sampler] WARNING: finalized {name} not present in "
                      f"{p} — its supply is excluded", file=sys.stderr)
        return files
    out = [Path(x) for x in sorted(glob.glob(triples))]
    # Explicit glob mode must not admit raw or usage intermediates.
    bad = [f for f in out if re.search(r"_raw|_usage", f.name)]
    if bad:
        sys.exit(f"[sampler] FATAL: glob matched raw/intermediate artifacts "
                 f"{[f.name for f in bad]} — pass the finalized directory or "
                 f"a stricter glob")
    return out


def count_pass(files: list[Path]) -> Counter:
    """Count eligible (row, intent) pairs per intent (contiguity dedup)."""
    counts = Counter()
    prev = None
    for fp in files:
        with open(fp) as f:
            for line in f:
                if _RX_LUNA.search(line) or '"answer": null' in line:
                    continue
                mi = _RX_INTENT.search(line)
                mr = _RX_ROW.search(line)
                if not mi or not mr:
                    continue
                key = (mr.group(1), mi.group(1))
                if key != prev:
                    counts[mi.group(1)] += 1
                    prev = key
    return counts


def waterfill(capacity: dict[str, int], total: int) -> dict[str, int]:
    """Distribute `total` across keys as evenly as possible, capped by
    per-key capacity. Deterministic (sorted keys)."""
    alloc = {k: 0 for k in capacity}
    remaining = total
    active = sorted(k for k in capacity if capacity[k] > 0)
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        nxt = []
        for k in active:
            take = min(share, capacity[k] - alloc[k], remaining)
            alloc[k] += take
            remaining -= take
            if alloc[k] < capacity[k]:
                nxt.append(k)
            if remaining <= 0:
                break
        if nxt == active and all(
                alloc[k] >= capacity[k] for k in capacity):
            break
        if not nxt:
            break
        active = [k for k in nxt if alloc[k] < capacity[k]]
    return alloc


def compute_targets(counts: Counter, n: int, held_frac: float) -> dict:
    """Per-intent sample targets honoring group quotas + held-out slice."""
    by_group = defaultdict(dict)
    from_grp = {}
    for iid, c in counts.items():
        grp = ("E/F" if iid[0] in "EF" else iid[0])
        from_grp[iid] = grp
        by_group[grp][iid] = c

    # held-out slice first
    held_cap = {i: counts.get(i, 0) for i in HELD_OUT_INTENTS}
    held_alloc = waterfill(held_cap, round(n * held_frac))
    n_train = n - sum(held_alloc.values())

    quotas = dict(GROUP_QUOTAS)
    missing = [g for g in quotas
               if not any(i for i in by_group.get(g, {})
                          if i not in HELD_OUT_INTENTS)]
    for g in missing:
        print(f"[sampler] WARNING: no supply for group {g} "
              f"(source not extracted yet) — quota redistributed",
              file=sys.stderr)
        quotas.pop(g)
    qsum = sum(quotas.values())

    targets = dict(held_alloc)
    shortfall = 0
    spare = {}   # intent -> spare capacity beyond target
    for g, q in sorted(quotas.items()):
        gt = round(n_train * q / qsum)
        cap = {i: c for i, c in by_group.get(g, {}).items()
               if i not in HELD_OUT_INTENTS}
        alloc = waterfill(cap, gt)
        got = sum(alloc.values())
        shortfall += gt - got
        targets.update(alloc)
        for i in cap:
            spare[i] = cap[i] - alloc[i]
    if shortfall > 0:
        extra = waterfill(spare, shortfall)
        for i, v in extra.items():
            targets[i] = targets.get(i, 0) + v
    # Independent per-group rounding can overshoot the requested total; trim
    # excess from the largest allocations without changing held-out targets.
    excess = sum(targets.values()) - n
    while excess > 0:
        big = max((i for i in targets if i not in HELD_OUT_INTENTS),
                  key=lambda i: targets[i], default=None)
        if big is None or targets[big] <= 0:
            break
        targets[big] -= 1
        excess -= 1
    return {i: t for i, t in targets.items() if t > 0}


def reservoir_sample(files: list[Path], targets: dict[str, int],
                     seed: int) -> dict[str, list[dict]]:
    """One streaming pass; per-intent reservoir of size targets[i], with
    one-triple-per-(row,intent) dedup (contiguous groups, seeded pick).

    Speed: reservoirs hold RAW LINES and only the final contents are
    json-parsed — the 67M-line full corpus never gets fully deserialized."""
    res = {i: [] for i in targets}
    seen = {i: 0 for i in targets}
    rngs = {i: random.Random(f"{seed}|res|{i}") for i in targets}
    pend_key, pend = None, []

    def flush():
        nonlocal pend_key, pend
        if not pend:
            return
        row, iid = pend_key
        pick = pend[0] if len(pend) == 1 else pend[
            random.Random(f"{seed}|pick|{iid}|{row}").randrange(len(pend))]
        seen[iid] += 1
        k = targets[iid]
        if len(res[iid]) < k:
            res[iid].append(pick)
        else:
            j = rngs[iid].randrange(seen[iid])
            if j < k:
                res[iid][j] = pick
        pend_key, pend = None, []

    for fp in files:
        with open(fp) as f:
            for line in f:
                mi = _RX_INTENT.search(line)
                if not mi or mi.group(1) not in targets:
                    continue
                if _RX_LUNA.search(line) or '"answer": null' in line:
                    continue
                mr = _RX_ROW.search(line)
                if not mr:
                    continue
                key = (mr.group(1), mi.group(1))
                if key != pend_key:
                    flush()
                    pend_key = key
                pend.append(line)
    flush()
    out = {}
    for iid, lines in res.items():
        parsed = [json.loads(x) for x in lines]
        out[iid] = [t for t in parsed if t.get("answer") is not None]
    return out


# --------------------------------------------------------------------------
# Corpus attachment
# --------------------------------------------------------------------------

def load_corpus_rows(corpus: str, idxs: set[int]) -> dict[int, dict]:
    from datasets import load_from_disk
    ds = load_from_disk(corpus)["train"]
    order = sorted(i for i in idxs if 0 <= i < len(ds))
    sub = ds.select(order)
    names = sub["func_name"]
    codes = sub["code"]
    quals = sub["qualified"]
    return {i: {"func_name": names[j].split(".")[-1], "code": codes[j],
                "qualified": quals[j]}
            for j, i in enumerate(order)}


def build_pair(t: dict, rows: dict[int, dict]) -> dict | None:
    """Triple -> self-contained sampled pair (code attached)."""
    seed_d = t["question_seed"]
    intent = t["intent"]
    stack_idxs = seed_d.get("stack_row_idxs")
    pid = pair_id_of(intent, t["row_idx"], seed_d)
    if stack_idxs:
        desc_idx = seed_d.get("describe_row_idx", t["row_idx"])
        if desc_idx not in rows or any(i not in rows for i in stack_idxs):
            return None
        code = rows[desc_idx]["code"]
        sibling_names = [rows[i]["func_name"] for i in stack_idxs]
        stack_code = "\n\n".join(rows[i]["code"] for i in stack_idxs)
        context = {"kind": "stack", "stack_row_idxs": stack_idxs,
                   "target_row_idx": desc_idx}
        # Leak-filter descriptions against every stack name. I5 names one
        # function by design, so only that seed name is exempt.
        _allowed = {seed_d.get("name")} if intent == "I5" else set()
        extra_leak = [n for n in sibling_names if n not in _allowed]
        needs_rt = "high"
    else:
        if t["row_idx"] not in rows:
            return None
        code = rows[t["row_idx"]]["code"]
        sibling_names, stack_code, extra_leak = None, None, []
        context = {"kind": "solo"}
        needs_rt = "normal"
    rec = {
        "pair_id": pid,
        "row_idx": t["row_idx"],
        "qualified": t["qualified"],
        "intent": intent,
        "group": t["group"],
        "family": INTENT_FAMILY[intent],
        "held_out": bool(t.get("held_out")),
        "answer": str(t["answer"]),
        "answer_type": map_answer_type(t["answer_type"], intent),
        "orig_answer_type": t["answer_type"],
        # Prefer explicit provenance; intent-based inference is only a fallback.
        "source": t.get("source") or infer_source(intent, seed_d),
        "context": context,
        "seed": seed_d,
        "code": code,
        "needs_roundtrip": needs_rt,
        "extra_leak_terms": extra_leak,
    }
    if "options" in seed_d:
        rec["options"] = seed_d["options"]
    if sibling_names:
        rec["sibling_names"] = sibling_names
        rec["stack_code"] = stack_code
    return rec


# --------------------------------------------------------------------------
# Stage: sample
# --------------------------------------------------------------------------

def stage_sample(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = triples_files(args.triples)
    if not files:
        sys.exit(f"no triples files under {args.triples}")
    # flush=True throughout: under nohup stdout is block-buffered and the
    # log stays empty for the whole hour-long pass otherwise
    print(f"[sample] {len(files)} triples files", flush=True)
    t0 = time.perf_counter()
    counts = count_pass(files)
    print(f"[sample] count pass: {sum(counts.values()):,} eligible "
          f"(row,intent) pairs in {time.perf_counter()-t0:.1f}s", flush=True)
    targets = compute_targets(counts, args.n, args.held_out_frac)
    t0 = time.perf_counter()
    res = reservoir_sample(files, targets, args.seed)
    print(f"[sample] reservoir pass in {time.perf_counter()-t0:.1f}s",
          flush=True)

    triples = [t for i in sorted(res) for t in
               sorted(res[i], key=lambda t: (t["row_idx"],
                                             json.dumps(t["question_seed"],
                                                        sort_keys=True)))]
    need = set()
    for t in triples:
        need.add(t["row_idx"])
        s = t["question_seed"]
        if "stack_row_idxs" in s:
            need.update(s["stack_row_idxs"])
            need.add(s.get("describe_row_idx", t["row_idx"]))
    print(f"[sample] loading {len(need):,} corpus rows for "
          f"{len(triples):,} sampled triples", flush=True)
    rows = load_corpus_rows(args.corpus, need)

    n_written, n_skipped = 0, 0
    sampled_counts = Counter()
    with open(out / "sampled_pairs.jsonl", "w") as f:
        for t in triples:
            rec = build_pair(t, rows)
            if rec is None:
                n_skipped += 1
                continue
            f.write(json.dumps(rec) + "\n")
            sampled_counts[rec["intent"]] += 1
            n_written += 1

    meta = {
        "n_requested": args.n, "seed": args.seed,
        "held_out_frac": args.held_out_frac,
        "triples_files": [str(f) for f in files],
        "supply": dict(counts), "targets": targets,
        "sampled": dict(sampled_counts),
        "row_missing_skipped": n_skipped,
    }
    with open(out / "sample_meta.json", "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print_quota_table(counts, targets, sampled_counts)
    print(f"[sample] wrote {n_written:,} pairs -> {out/'sampled_pairs.jsonl'}"
          f" ({n_skipped} skipped for missing corpus rows)", flush=True)


def print_quota_table(counts, targets, sampled):
    groups = defaultdict(lambda: [0, 0, 0])
    print(f"{'intent':8} {'supply':>12} {'target':>9} {'sampled':>9}")
    for i in sorted(targets):
        print(f"{i:8} {counts.get(i, 0):>12,} {targets[i]:>9,} "
              f"{sampled.get(i, 0):>9,}")
        g = "E/F" if i[0] in "EF" else i[0]
        groups[g][0] += counts.get(i, 0)
        groups[g][1] += targets[i]
        groups[g][2] += sampled.get(i, 0)
    tot = [0, 0, 0]
    print("-- groups --")
    for g in sorted(groups):
        s, t, m = groups[g]
        share = m / max(1, sum(v[2] for v in groups.values()))
        print(f"{g:8} {s:>12,} {t:>9,} {m:>9,}  ({share:.1%} of sampled; "
              f"quota {GROUP_QUOTAS.get(g, 0):.0%})")
        for k in range(3):
            tot[k] += groups[g][k]
    print(f"{'TOTAL':8} {tot[0]:>12,} {tot[1]:>9,} {tot[2]:>9,}")


# --------------------------------------------------------------------------
# Stage: phrase (mock + real transport)
# --------------------------------------------------------------------------

_MOCK_DESCS = [
    "performs its documented transformation on the inputs",
    "handles one well-defined piece of this file's work",
    "computes a derived value from its arguments",
    "carries out a specific bookkeeping step",
]


def mock_question(rec: dict, seed: int) -> tuple[str, str]:
    """Deterministic template fill; returns (question, template_id)."""
    intent = rec["intent"]
    templates = INTENT_TASKS[intent][1]
    rng = random.Random(f"{seed}|mock|{rec['pair_id']}")
    ti = rng.randrange(len(templates))
    fields = defaultdict(str, {k: v for k, v in rec["seed"].items()
                               if isinstance(v, (str, int, float))})
    for key, dst in (("n", "nth"), ("k", "kth")):
        if key in rec["seed"]:
            fields[dst] = _ordinal(rec["seed"][key])
    if intent == "C4":
        fields["property"] = _c4_phrase(rec["seed"])
    if intent == "E1":
        fields["exit_kind"] = _e1_kind(rec["seed"])
    if intent.startswith("I"):
        fields["mock_desc"] = _MOCK_DESCS[rng.randrange(len(_MOCK_DESCS))]
        if intent == "I4":
            inner = rec["seed"].get("inner", {"intent": "A2"})
            inner_t = INTENT_TASKS[inner["intent"]][1][0]
            fields["mock_inner"] = inner_t.format_map(
                defaultdict(str, inner)).rstrip("?").lower() + "?"
    q = templates[ti].format_map(fields)
    return q, f"{intent}.mock{ti}"


def postprocess_llm(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.strip().strip("`").strip()
    for pfx in ("Question:", "question:", "Q:"):
        if text.startswith(pfx):
            text = text[len(pfx):].strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return " ".join(text.split())


async def llm_phrase(records: list[dict], args, out_path: Path,
                     done_path: Path) -> Counter:
    """Massive-concurrency phrasing against VLLM_URL. Appends results to
    out_path as they complete; done ids to done_path (resume support)."""
    import aiohttp

    url = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B")
    sem = asyncio.Semaphore(args.concurrency)
    stats = Counter()
    out_f = open(out_path, "a")
    done_f = open(done_path, "a")
    lock = asyncio.Lock()

    async def one(session, rec):
        sid, directive = choose_style(rec["pair_id"], rec["intent"],
                                      args.seed)
        payload = {
            "model": model,
            "messages": build_messages(rec, directive),
            "temperature": args.temperature,
            "max_tokens": 160,
            # Qwen3: disable thinking for short surface-form generation
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with sem:
            for attempt in range(6):
                try:
                    async with session.post(
                            f"{url}/chat/completions", json=payload,
                            timeout=aiohttp.ClientTimeout(total=180)) as r:
                        if r.status in (429, 500, 502, 503, 504):
                            raise aiohttp.ClientError(f"HTTP {r.status}")
                        r.raise_for_status()
                        data = await r.json()
                    q = postprocess_llm(
                        data["choices"][0]["message"]["content"])
                    if not q:
                        raise ValueError("empty generation")
                    rec2 = dict(rec)
                    rec2.update(question=q, style_id=sid, mock=False,
                                planted_leak=False, gen_model=model)
                    async with lock:
                        out_f.write(json.dumps(rec2) + "\n")
                        done_f.write(rec["pair_id"] + "\n")
                        out_f.flush(); done_f.flush()
                    stats["ok"] += 1
                    return
                except (aiohttp.ClientError, asyncio.TimeoutError,
                        ValueError, KeyError) as e:
                    if attempt == 5:
                        stats["failed"] += 1
                        stats[f"fail:{type(e).__name__}"] += 1
                        return
                    await asyncio.sleep(min(60, 2 ** attempt)
                                        + random.random())

    async with aiohttp.ClientSession() as session:
        chunk = 20000  # bound task-list memory
        for i in range(0, len(records), chunk):
            await asyncio.gather(*(one(session, r)
                                   for r in records[i:i + chunk]))
            print(f"[phrase] {min(i+chunk, len(records))}/{len(records)} "
                  f"done ({stats['ok']} ok, {stats['failed']} failed)",
                  flush=True)
    out_f.close(); done_f.close()
    return stats


def stage_phrase(args) -> None:
    out = Path(args.out_dir)
    src = out / "sampled_pairs.jsonl"
    if not src.exists():
        sys.exit(f"{src} missing — run the sample stage first")
    records = []
    with open(src) as f:
        for line in f:
            records.append(json.loads(line))
    if args.limit:
        # stratified round-robin across intents, so a --limit 300 mini-eval
        # sees every intent, not just the alphabetically first one
        by_intent = defaultdict(list)
        for r in records:
            by_intent[r["intent"]].append(r)
        order = sorted(by_intent)
        picked, i = [], 0
        while len(picked) < args.limit and any(by_intent.values()):
            iid = order[i % len(order)]
            if by_intent[iid]:
                picked.append(by_intent[iid].pop(0))
            i += 1
        records = picked
    done_path = out / "phrase_done.ids"
    done = set()
    if done_path.exists():
        done = set(done_path.read_text().split())
    todo = [r for r in records if r["pair_id"] not in done]
    print(f"[phrase] {len(records):,} sampled, {len(done):,} already done, "
          f"{len(todo):,} to phrase (mock={args.mock})")

    out_path = out / "qa_raw.jsonl"
    if args.mock:
        plant_ids = choose_plant_ids(todo, args.plant_leaks, args.seed)
        with open(out_path, "a") as f, open(done_path, "a") as df:
            for rec in todo:
                q, tid = mock_question(rec, args.seed)
                sid, _ = choose_style(rec["pair_id"], rec["intent"],
                                      args.seed)
                planted = rec["pair_id"] in plant_ids
                if planted:
                    q = f"{q.rstrip('?')} — is it {rec['answer']}?"
                rec2 = dict(rec)
                rec2.update(question=q, style_id=sid, template_id=tid,
                            mock=True, planted_leak=planted,
                            gen_model="mock")
                f.write(json.dumps(rec2) + "\n")
                df.write(rec["pair_id"] + "\n")
        print(f"[phrase] mock-phrased {len(todo):,} "
              f"({len(plant_ids)} planted leaks) -> {out_path}")
    else:
        stats = asyncio.run(llm_phrase(todo, args, out_path, done_path))
        print(f"[phrase] {dict(stats)} -> {out_path}")
        if stats["failed"] > 0:
            # Fail closed so incomplete requests cannot silently bias filtering.
            # Rerunning resumes from phrase_done.ids and retries only failures.
            sys.exit(f"[phrase] FATAL: {stats['failed']} requests exhausted "
                     f"retries — rerun this command to retry them before "
                     f"filtering")


def choose_plant_ids(records, k, seed) -> set[str]:
    """Deterministically pick k records whose answers the leak filter can
    catch (skip yesno/letter — filter ignores those by design)."""
    elig = [r["pair_id"] for r in records
            if r["answer_type"] not in ("yesno", "letter")
            and 1 <= len(str(r["answer"])) <= 120
            and str(r["answer"]).lower() not in TRIVIAL_LEAK_EXEMPT]
    rng = random.Random(f"{seed}|plant")
    rng.shuffle(elig)
    return set(elig[:k])


# --------------------------------------------------------------------------
# Stage: render-templates
# --------------------------------------------------------------------------

def stage_render(args) -> None:
    out = Path(args.templates_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# QA phrasing — prompt & template bank\n")
    L.append("Rendered by `dataset/qa_phrase.py render-templates` — edit the "
             "source, not this file.\n")
    L.append("Generation prompt structure: SYSTEM below + a user message "
             "carrying (code block, fact card, [MCQ options], ground-truth "
             "answer, style directive). One style directive is sampled per "
             "call, seeded by pair_id.\n")
    L.append("## System prompt\n\n```\n" + SYSTEM_PROMPT + "\n```\n")
    L.append("## Per-record addenda\n")
    L.append("- MCQ records append:\n```" + MCQ_ADDENDUM + "\n```")
    L.append("- Stack (group I, except I6) records append:\n```"
             + STACK_ADDENDUM + "\n```")
    L.append("- I6 (multi-match) records append:\n```" + I6_STACK_ADDENDUM
             + "\n```\n")
    L.append("## Style directives per family "
             f"({len(BASE_STYLES)} shared + 4 family-specific = "
             f"{len(BASE_STYLES)+4} each)\n")
    L.append("### Shared base styles (all families)\n")
    for sid, d in BASE_STYLES:
        L.append(f"- **{sid}** — {d}")
    for fam in sorted(FAMILY_EXTRA_STYLES):
        L.append(f"\n### Family `{fam}` extras\n")
        for sid, d in FAMILY_EXTRA_STYLES[fam]:
            L.append(f"- **{sid}** — {d}")
        members = sorted(i for i, f in INTENT_FAMILY.items() if f == fam)
        L.append(f"\nIntents: {', '.join(members)}")
    L.append("\n## Fact cards per intent (rendered with a representative "
             "seed) + mock templates\n")
    REP_SEEDS = {
        "A4": {"param": "timeout"}, "A5": {"feature": "*args"},
        "A7": {"param": "path"}, "A10": {"aspect": "is_method"},
        "A11": {"count_of": "loops"}, "A12": {"n": 2},
        "B1": {"line": 4}, "B2": {"construct": "if", "k": 1},
        "B3": {"k": 2}, "B5": {"construct": "for-loop body", "k": 1},
        "B6": {"construct": "with statement"},
        "C2": {"param": "items", "situation": "empty"},
        "C4": {"property": "recursive"}, "C7": {"var": "total"},
        "D1": {"input": "f(3, 4)"}, "D2": {"output": "'ok'"},
        "D3": {"var": "n", "after_line": 5, "input": "f([1, 2])"},
        "D4": {"line": 6, "input": "f(0)"},
        "D5": {"if_line": 3, "input": "f(-1)"},
        "D6": {"loop_line": 4, "input": "f('abc')"},
        "D7": {"input": "f(None)"},
        "E1": {"exit_kind": "return None"}, "F1": {"var": "result"},
        "G3": {"aspect": "which"}, "H3": {"aspect": "style"},
        "I5": {"name": "parse_header"},
    }
    for iid in sorted(INTENT_TASKS, key=lambda x: (x[0], int(x[1:]))):
        task_fn, mocks = INTENT_TASKS[iid]
        seed_d = REP_SEEDS.get(iid, {})
        L.append(f"### {iid} ({INTENT_FAMILY[iid]})\n")
        L.append(f"Fact card: {task_fn(seed_d)}\n")
        L.append("Mock templates (pipeline testing only, never trained):")
        for m in mocks:
            L.append(f"- `{m}`")
        L.append("")
    L.append("## Notes\n")
    L.append("- LUNA-answered triples (C1, C3, C2-not-runnable) are phrased "
             "like any other source once finalized triples_luna.jsonl exists "
             "(answers were frontier-authored upstream; only the question is "
             "written here).")
    L.append("- I-group records are marked needs_roundtrip=high: the model "
             "authors the behavioral description, so round-trip "
             "disambiguation is load-bearing, not optional.")
    L.append("- I6 (multi-match) supply comes from triples_i6.jsonl "
             "(dataset/qa_i6_multimatch.py) and uses its own stack addendum "
             "(exactly-two-match, never single-out).")
    out.write_text("\n".join(L) + "\n")
    n_styles = {f: len(FAMILY_STYLES[f]) for f in FAMILY_STYLES}
    print(f"[render] {out} written; styles per family: {n_styles}; "
          f"{len(INTENT_TASKS)} intent fact cards")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["sample", "phrase", "all",
                                      "render-templates"])
    ap.add_argument("--triples", default="data/corpus_v2/qa_full")
    ap.add_argument("--corpus", default="data/corpus_v2/corpus")
    ap.add_argument("--out-dir", default="data/corpus_v2/qa_phrased")
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--held-out-frac", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0,
                    help="phrase only the first K sampled pairs (mini-eval)")
    ap.add_argument("--mock", action="store_true",
                    help="deterministic template fill, no LLM/network")
    ap.add_argument("--plant-leaks", type=int, default=0,
                    help="mock only: inject K deliberate answer leaks")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--templates-out",
                    default=str(REPO / "out/qa/PHRASING_TEMPLATES.md"))
    args = ap.parse_args()

    if args.stage == "render-templates":
        stage_render(args)
        return
    if args.stage in ("sample", "all"):
        stage_sample(args)
    if args.stage in ("phrase", "all"):
        stage_phrase(args)


if __name__ == "__main__":
    main()
