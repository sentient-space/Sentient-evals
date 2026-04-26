from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ...artifacts import TrialArtifacts
from ...models import GraderResult, Severity, Task, TranscriptEvent


_AllowedRole = str  # "system" | "user" | "assistant" — kept loose for forward compat


def _collect_text(
    transcript: Sequence[TranscriptEvent],
    *,
    roles: tuple[str, ...] | None,
    include_reasoning: bool,
) -> str:
    parts: list[str] = []
    for event in transcript:
        if event.kind != "message":
            continue
        if roles and (event.role is None or event.role not in roles):
            continue
        if event.content:
            parts.append(event.content)
        if include_reasoning and event.reasoning_content:
            parts.append(event.reasoning_content)
    return "\n".join(parts)


def _match(pattern: str, text: str, *, regex: bool, case_sensitive: bool) -> bool:
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            return re.search(pattern, text, flags) is not None
        except re.error:
            return False
    if case_sensitive:
        return pattern in text
    return pattern.lower() in text.lower()


@dataclass(frozen=True)
class TranscriptGrader:
    """Pattern-based grader that scans the transcript text for required and forbidden phrases.

    Config keys:
      - must_contain: list[str] — every pattern must appear at least once.
      - must_not_contain: list[str] — none of these may appear.
      - regex: bool — interpret patterns as regular expressions.
      - case_sensitive: bool — default False.
      - roles: list[str] | None — filter to these roles (e.g. ["assistant"]).
      - include_reasoning: bool — also scan reasoning_content alongside content.
    """

    name: str = "transcript"
    must_contain: tuple[str, ...] = field(default_factory=tuple)
    must_not_contain: tuple[str, ...] = field(default_factory=tuple)
    regex: bool = False
    case_sensitive: bool = False
    roles: tuple[str, ...] | None = None
    include_reasoning: bool = False

    async def grade(
        self,
        *,
        task: Task,
        transcript: Sequence[TranscriptEvent],
        outcome: Any,
        artifacts: TrialArtifacts,
    ) -> GraderResult:
        text = _collect_text(
            transcript,
            roles=self.roles,
            include_reasoning=self.include_reasoning,
        )

        missing: list[str] = [
            pat
            for pat in self.must_contain
            if not _match(pat, text, regex=self.regex, case_sensitive=self.case_sensitive)
        ]
        forbidden_hit: list[str] = [
            pat
            for pat in self.must_not_contain
            if _match(pat, text, regex=self.regex, case_sensitive=self.case_sensitive)
        ]

        passed = not missing and not forbidden_hit
        return GraderResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            severity=Severity.info if passed else Severity.error,
            details={
                "missing": missing,
                "forbidden_hit": forbidden_hit,
                "scanned_chars": len(text),
                "regex": self.regex,
                "roles": list(self.roles) if self.roles else None,
            },
        )


def build_transcript_grader(cfg: dict[str, Any]) -> TranscriptGrader:
    must_contain = cfg.get("must_contain") or []
    must_not_contain = cfg.get("must_not_contain") or []
    roles = cfg.get("roles")
    if not isinstance(must_contain, (list, tuple)):
        raise ValueError("transcript grader: must_contain must be a list of strings")
    if not isinstance(must_not_contain, (list, tuple)):
        raise ValueError("transcript grader: must_not_contain must be a list of strings")
    if roles is not None and not isinstance(roles, (list, tuple)):
        raise ValueError("transcript grader: roles must be a list of strings or null")
    return TranscriptGrader(
        must_contain=tuple(str(p) for p in must_contain),
        must_not_contain=tuple(str(p) for p in must_not_contain),
        regex=bool(cfg.get("regex", False)),
        case_sensitive=bool(cfg.get("case_sensitive", False)),
        roles=tuple(str(r) for r in roles) if roles else None,
        include_reasoning=bool(cfg.get("include_reasoning", False)),
    )
