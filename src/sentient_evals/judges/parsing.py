from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_SCALE_MAX = {
    "0-1": 1.0,
    "binary": 1.0,
    "1-5": 5.0,
    "1-10": 10.0,
}


@dataclass(frozen=True)
class ParsedJudgeResponse:
    verdict: str
    score: float
    normalized_score: float
    reason: str
    payload: dict[str, Any]


def score_scale_max(score_scale: str | None) -> float:
    return _SCALE_MAX.get(str(score_scale or "0-1").strip().lower(), 1.0)


def score_scale_label(score_scale: str | None) -> str:
    scale = str(score_scale or "0-1").strip().lower()
    if scale == "binary":
        return "0 or 1"
    if scale == "1-5":
        return "1 to 5"
    if scale == "1-10":
        return "1 to 10"
    return "0 to 1"


def normalize_score(score: float, score_scale: str | None) -> float:
    max_score = score_scale_max(score_scale)
    return max(0.0, min(1.0, float(score) / max_score))


def _strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    match = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```\s*$", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    parts = stripped.split("```")
    if len(parts) >= 3:
        candidate = parts[1].strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
        return candidate
    return stripped


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_payload(text: str) -> dict[str, Any]:
    candidates = [str(text or "").strip()]
    fenced = _strip_markdown_json_fence(str(text or ""))
    if fenced not in candidates:
        candidates.append(fenced)
    embedded = _first_json_object(fenced)
    if embedded and embedded not in candidates:
        candidates.append(embedded)

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except Exception as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            last_error = ValueError("judge response JSON must be an object")
            continue
        return payload
    raise ValueError(f"Could not parse judge response JSON: {last_error}")


def parse_judge_response(
    text: str,
    *,
    score_scale: str | None = "0-1",
    default_verdict: str = "unknown",
) -> ParsedJudgeResponse:
    payload = extract_json_payload(text)
    verdict = str(payload.get("verdict") or default_verdict).strip().lower() or default_verdict
    raw_score = payload.get("score", 0.0)
    try:
        score = float(raw_score)
    except Exception as exc:
        raise ValueError(f"judge response score must be numeric: {raw_score!r}") from exc
    max_score = score_scale_max(score_scale)
    score = max(0.0, min(max_score, score))
    reason = str(payload.get("reason", payload.get("reasoning", "")))
    return ParsedJudgeResponse(
        verdict=verdict,
        score=score,
        normalized_score=normalize_score(score, score_scale),
        reason=reason,
        payload=payload,
    )

