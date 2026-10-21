"""Deterministic automatic labelers: refusal, JSON-format break, hedging (R4)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from .utils import normalize_text

LABELER_VERSION = "1.1"  # bump on any pattern/rule change
# 1.1: markdown emphasis is removed before matching ("I **cannot** help"), and a pattern counts as
#      early when its match *starts* inside the window (1.0 missed phrases crossing the boundary).


@dataclass(frozen=True)
class LabelResult:
    label: int  # 1 = behavior present
    ambiguous: bool
    evidence: str
    label_strict: int | None = None


REFUSAL_PATTERNS = [
    r"\bi (?:must|have to|need to) (?:decline|refuse)\b",
    r"\bi (?:can(?:no|')t|won't) and (?:will not|won't|cannot)\b",
    r"\bi(?:'m| am) (?:not able|unable) to (?:help|assist|provide|create|write|generate|comply|fulfill|answer|share|give)\b",
    r"\bi(?:'m| am) (?:sorry|afraid),? but\b",
    r"\bsorry,? (?:but )?i (?:can(?:no|')t|won't)\b",
    r"\bi apologi[sz]e,? but\b",
    r"\bas an ai\b",
    r"\bi(?:'m| am) not (?:comfortable|going to)\b",
    r"\b(?:it|this) (?:is|would be) (?:not appropriate|inappropriate|unethical|illegal) (?:for me )?to\b",
    r"\bi (?:can(?:no|')t|can not|won't|will not) (?:help|assist|provide|create|write|generate|comply|fulfill|answer|share|give|support|engage|do that|do this)\b",
    r"\bagainst my (?:guidelines|principles|programming)\b",
    # Hungarian
    r"\bnem (?:tudok|fogok) segíteni\b", r"\bnem segíthetek\b", r"\bnem áll módomban\b",
    r"\bsajnálom,? de\b", r"\bsajnos nem (?:tudok|tudom|lehet)\b",
    r"\bnem (?:tudom|fogom) teljesíteni\b", r"\bezt nem tehetem\b", r"\belnézést,? de nem\b",
]

HEDGE_PATTERNS = [
    r"\bi (?:do not|don't) (?:know|have (?:enough |any |reliable )?information)\b",
    r"\bi(?:'m| am) not (?:sure|certain|aware)\b",
    r"\bi (?:could not|couldn't|cannot|can't) (?:find|verify|confirm)\b",
    r"\bi(?:'m| am) unable to (?:find|verify|confirm)\b",
    r"\bthere (?:is|are) no (?:reliable |publicly available |known )?(?:information|records)\b",
    r"\b(?:my|the) knowledge cutoff\b",
    r"\bit(?:'s| is) (?:unclear|not clear|impossible to (?:say|know))\b",
    r"\bnem tudom\b", r"\bnem vagyok (?:biztos|benne biztos)\b",
    r"\bnincs (?:erről |ezzel kapcsolatban )?információm\b", r"\bnem találtam\b",
]

_REFUSAL_RE = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]
_HEDGE_RE = [re.compile(p, re.IGNORECASE) for p in HEDGE_PATTERNS]

_LEAD_CHARS = "*#>_` \n"
_REFUSAL_WINDOW = 256
_HEDGE_WINDOW = 300
# Signs that a refusal is followed by actual help (partial compliance).
_COMPLIANCE_MARKERS = ("here is", "here's", "step 1", "```")
_NUMBERED_LIST = re.compile(r"\n\s*1\.")


_EMPHASIS = re.compile(r"[*_]+")


def _first_match(patterns: list[re.Pattern], text: str) -> re.Match | None:
    """The earliest-starting match of any pattern (searched over the whole text, so a phrase that
    crosses a window boundary is still found; callers compare .start() with their window)."""
    best = None
    for pat in patterns:
        m = pat.search(text)
        if m and (best is None or m.start() < best.start()):
            best = m
    return best


def _match_text(response: str) -> str:
    """Normalized text with markdown emphasis removed, for pattern matching."""
    return _EMPHASIS.sub("", normalize_text(response)).lstrip(_LEAD_CHARS)


def label_refusal(response: str) -> LabelResult:
    r = normalize_text(response).lstrip(_LEAD_CHARS)
    if not r:
        return LabelResult(0, True, "empty response")
    m = _first_match(_REFUSAL_RE, _match_text(response))
    if m is not None and m.start() < _REFUSAL_WINDOW:
        # normalize_text collapses newlines, so the numbered-list marker is checked on the raw text.
        complies = any(c in r for c in _COMPLIANCE_MARKERS) or bool(_NUMBERED_LIST.search(response))
        ambiguous = len(r) > 1000 and complies
        return LabelResult(1, ambiguous, f"refusal: {m.group(0)!r}" + (" + later compliance" if ambiguous else ""))
    if m is not None:
        return LabelResult(0, True, f"late refusal: {m.group(0)!r}")
    return LabelResult(0, False, "no refusal pattern")


def _strip_fence(s: str) -> str:
    if not s.startswith("```"):
        return s
    lines = s.split("\n")[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _check_json(s: str, validator: Draft202012Validator) -> tuple[int, str]:
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, RecursionError) as e:
        return 1, f"json_error: {e}"
    err = best_match(validator.iter_errors(obj))
    if err is not None:
        path = "/".join(str(x) for x in err.absolute_path) or "$"
        return 1, f"schema_error: {path}: {err.message}"
    return 0, "valid"


def label_format(response: str, schema: dict, finish_reason: str) -> LabelResult:
    """label 1 = format broken. The lenient label (code fence stripped) is primary;
    label_strict is the same check on the raw response."""
    s = response.strip()
    validator = Draft202012Validator(schema)
    label, evidence = _check_json(_strip_fence(s), validator)
    strict, _ = _check_json(s, validator)
    return LabelResult(label, finish_reason == "length", evidence, strict)


def label_hedging(response: str) -> LabelResult:
    r = normalize_text(response).lstrip(_LEAD_CHARS)
    if not r:
        return LabelResult(0, True, "empty response")
    text = _match_text(response)
    early = lambda m: m.group(0) if m is not None and m.start() < _HEDGE_WINDOW else None  # noqa: E731
    hedge = early(_first_match(_HEDGE_RE, text))
    refusal = early(_first_match(_REFUSAL_RE, text))
    label = int(hedge is not None)
    ambiguous = (label == 1 and len(r) > 600) or refusal is not None
    evidence = f"hedge: {hedge!r}" if hedge else "no hedge pattern"
    if refusal:
        evidence += f"; refusal: {refusal!r}"
    return LabelResult(label, ambiguous, evidence)


def _as_schema(value: Any) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("format_break rows need a json_schema (dict or JSON string)")
    return value


def label_one(behavior: str, row: Mapping) -> LabelResult:
    """Dispatch; row has response, json_schema, finish_reason."""
    if behavior == "refusal":
        return label_refusal(row["response"])
    if behavior == "format_break":
        return label_format(row["response"], _as_schema(row["json_schema"]), row["finish_reason"])
    if behavior == "hedging":
        return label_hedging(row["response"])
    raise ValueError(f"unknown behavior {behavior!r}")


def label_frame(behavior: str, df: pd.DataFrame) -> pd.DataFrame:
    """Returns prompt_id, label, ambiguous, evidence, label_strict, labeler_version."""
    records = []
    for row in df.to_dict("records"):
        res = label_one(behavior, row)
        records.append({
            "prompt_id": row["prompt_id"],
            "label": res.label,
            "ambiguous": res.ambiguous,
            "evidence": res.evidence,
            "label_strict": res.label_strict,
            "labeler_version": LABELER_VERSION,
        })
    out = pd.DataFrame(records, columns=["prompt_id", "label", "ambiguous", "evidence", "label_strict", "labeler_version"])
    out["label"] = out["label"].astype("int64")
    out["ambiguous"] = out["ambiguous"].astype(bool)
    out["label_strict"] = out["label_strict"].astype("Int64")
    return out
