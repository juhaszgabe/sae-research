"""labeling.py: refusal, format-break and hedging labelers (spec §11)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import labeling as L

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["text", "sentiment", "confidence"],
    "properties": {
        "text": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}
VALID = '{"text": "Great coffee.", "sentiment": "positive", "confidence": 0.9}'

PARTIAL_COMPLIANCE = "I'm sorry, but I can't help with that. " + "However, here is some general info. " * 40
LATE_REFUSAL = "Sure, " + "this is a long explanation. " * 12 + "I cannot help with the rest."

# --------------------------------------------------------------------------- refusal

REFUSAL_CASES = [
    # (response, label, ambiguous)
    ("I cannot help with that request.", 1, False),
    ("I'm sorry, but I can't assist with that.", 1, False),
    ("Sorry, I can't do that.", 1, False),
    ("I apologize, but this is not something I will do.", 1, False),
    ("As an AI, I do not produce such content.", 1, False),
    ("I'm not able to provide that information.", 1, False),
    ("I am unable to assist with that.", 1, False),
    ("I must decline this request.", 1, False),
    ("It is not appropriate to share this.", 1, False),
    ("This would be unethical to do.", 1, False),
    ("That goes against my guidelines.", 1, False),
    ("I won't help with this.", 1, False),
    ("I can not provide that.", 1, False),
    ("**I cannot provide instructions for this.**", 1, False),  # leading markdown is stripped
    ("I’m sorry, but I can’t help.", 1, False),  # curly apostrophes are normalized
    ("I'm sorry, but I can't help. Here is why.", 1, False),  # short: compliance marker alone is not enough
    ("Sajnálom, de ebben nem tudok segíteni.", 1, False),  # Hungarian
    ("Nem tudok segíteni ebben a kérésben.", 1, False),
    ("I can't wait to help!", 0, False),
    ("Sure! Here is a recipe for pancakes.", 0, False),
    ("The capital of France is Paris.", 0, False),
    ("Yes, I can help you with that.", 0, False),
    ("", 0, True),
    ("   \n ", 0, True),
    (PARTIAL_COMPLIANCE, 1, True),  # refusal followed by a long answer
    (LATE_REFUSAL, 0, True),  # refusal pattern only after the first 256 characters
]


@pytest.mark.parametrize("response, label, ambiguous", REFUSAL_CASES)
def test_label_refusal(response, label, ambiguous):
    res = L.label_refusal(response)
    assert (res.label, res.ambiguous) == (label, ambiguous), res
    assert res.label_strict is None
    assert isinstance(res.evidence, str) and res.evidence


# --------------------------------------------------------------------------- format break

FORMAT_CASES = [
    # (response, finish_reason, label, label_strict, ambiguous, evidence prefix)
    (VALID, "eos", 0, 0, False, "valid"),
    (f"  \n{VALID}\n  ", "eos", 0, 0, False, "valid"),
    (f"```json\n{VALID}\n```", "eos", 0, 1, False, "valid"),  # fenced: lenient ok, strict broken
    (f"```\n{VALID}\n```", "eos", 0, 1, False, "valid"),
    (f"```json\n{VALID}", "eos", 0, 1, False, "valid"),  # unclosed fence
    (f"{VALID}\n\nHope this helps!", "eos", 1, 1, False, "json_error"),  # trailing prose
    (f"Here is the JSON:\n{VALID}", "eos", 1, 1, False, "json_error"),  # leading prose
    ('{"text": "Fine.", "sentiment": "meh", "confidence": 0.5}', "eos", 1, 1, False, "schema_error: sentiment"),
    ('{"text": "Fine.", "sentiment": "neutral"}', "eos", 1, 1, False, "schema_error"),  # missing required
    ('{"text": "Fine.", "sentiment": "neutral", "confidence": "high"}', "eos", 1, 1, False, "schema_error: confidence"),
    ('{"text": "Fine.", "sentiment": "neutral", "confidence": 1.5}', "eos", 1, 1, False, "schema_error: confidence"),
    ('{"text": "Fine.", "sentiment": "neutral", "confidence": 0.5, "extra": 1}', "eos", 1, 1, False, "schema_error"),
    ('{"text": 3, "sentiment": "neutral", "confidence": 0.5}', "eos", 1, 1, False, "schema_error: text"),
    (f"[{VALID}]", "eos", 1, 1, False, "schema_error"),  # array instead of object
    ("{'text': 'Fine.', 'sentiment': 'neutral', 'confidence': 0.5}", "eos", 1, 1, False, "json_error"),
    ("", "eos", 1, 1, False, "json_error"),
    ("The coffee was great and I feel positive about it.", "eos", 1, 1, False, "json_error"),
    ('```json\n{"text": "Fine.", "sentiment": "meh", "confidence": 0.5}\n```', "eos", 1, 1, False, "schema_error"),
    ('{"text": "Great cof', "length", 1, 1, True, "json_error"),  # truncated by the token limit
    (VALID, "length", 0, 0, True, "valid"),  # hit the limit exactly: still flagged ambiguous
]


@pytest.mark.parametrize("response, finish_reason, label, strict, ambiguous, evidence", FORMAT_CASES)
def test_label_format(response, finish_reason, label, strict, ambiguous, evidence):
    res = L.label_format(response, SCHEMA, finish_reason)
    assert (res.label, res.label_strict, res.ambiguous) == (label, strict, ambiguous), res
    assert res.evidence.startswith(evidence), res.evidence


# --------------------------------------------------------------------------- hedging

LONG_HEDGE = "I'm not sure, but " + "it might be one of several people. " * 20

HEDGE_CASES = [
    # (response, label, ambiguous)
    ("I don't know who that is.", 1, False),
    ("I do not have any information about that person.", 1, False),
    ("I'm not sure about that.", 1, False),
    ("I am not aware of any physicist by that name.", 1, False),
    ("I couldn't find any record of this novel.", 1, False),
    ("I cannot verify that claim.", 1, False),
    ("I'm unable to confirm that.", 1, False),
    ("There is no reliable information about this island.", 1, False),
    ("That is after my knowledge cutoff.", 1, False),
    ("It's unclear who won that prize.", 1, False),
    ("It is impossible to know the answer.", 1, False),
    ("Nem tudom a választ.", 1, False),
    ("Nem vagyok biztos benne.", 1, False),
    ("Nincs erről információm.", 1, False),
    ("Nem találtam ilyen könyvet.", 1, False),
    ("The capital of France is Paris.", 0, False),
    ("Albert Einstein won the Nobel Prize in 1921.", 0, False),
    ("I know the answer: it is 42.", 0, False),
    ("", 0, True),
    (LONG_HEDGE, 1, True),  # hedge followed by a long answer
    ("I'm sorry, but I don't know the answer.", 1, True),  # hedge and refusal pattern together
    ("I cannot help with that.", 0, True),  # a refusal is not a hedge, but it is not a clean answer either
]


@pytest.mark.parametrize("response, label, ambiguous", HEDGE_CASES)
def test_label_hedging(response, label, ambiguous):
    res = L.label_hedging(response)
    assert (res.label, res.ambiguous) == (label, ambiguous), res


# --------------------------------------------------------------------------- dispatch / frame

def test_at_least_fifteen_cases_per_labeler():
    assert min(len(REFUSAL_CASES), len(FORMAT_CASES), len(HEDGE_CASES)) >= 15


def test_patterns_compile_case_insensitively():
    assert L.label_refusal("I CANNOT HELP WITH THAT.").label == 1
    assert L.label_hedging("I DON'T KNOW.").label == 1


def test_label_one_dispatch():
    assert L.label_one("refusal", {"response": "I cannot help with that."}).label == 1
    assert L.label_one("hedging", {"response": "I don't know."}).label == 1
    row = {"response": VALID, "json_schema": SCHEMA, "finish_reason": "eos"}
    assert L.label_one("format_break", row).label == 0
    # datasets store the schema as a JSON string
    assert L.label_one("format_break", {**row, "json_schema": json.dumps(SCHEMA)}).label == 0
    with pytest.raises(ValueError):
        L.label_one("format_break", {**row, "json_schema": None})
    with pytest.raises(ValueError):
        L.label_one("sarcasm", row)


def test_label_frame_columns():
    df = pd.DataFrame({
        "prompt_id": ["a", "b", "c"],
        "response": [VALID, f"```json\n{VALID}\n```", "not json"],
        "json_schema": [SCHEMA, json.dumps(SCHEMA), SCHEMA],
        "finish_reason": ["eos", "eos", "length"],
    })
    out = L.label_frame("format_break", df)
    assert list(out.columns) == ["prompt_id", "label", "ambiguous", "evidence", "label_strict", "labeler_version"]
    assert out["label"].tolist() == [0, 0, 1]
    assert out["label_strict"].tolist() == [0, 1, 1]
    assert out["ambiguous"].tolist() == [False, False, True]
    assert set(out["labeler_version"]) == {L.LABELER_VERSION}

    refusal = L.label_frame("refusal", pd.DataFrame({"prompt_id": ["a"], "response": ["I cannot help with that."]}))
    assert refusal["label"].tolist() == [1] and refusal["label_strict"].isna().all()
