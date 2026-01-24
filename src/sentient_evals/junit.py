from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from xml.etree.ElementTree import Element, SubElement, tostring

from .models import TrialResult


@dataclass(frozen=True)
class JUnitExportConfig:
    suite_name: str = "sentient-evals"


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def trials_to_junit_xml(trials: Iterable[TrialResult], cfg: JUnitExportConfig) -> str:
    trials_list = list(trials)
    testsuite = Element("testsuite", {"name": cfg.suite_name, "tests": str(len(trials_list))})

    failures = 0
    errors = 0
    for t in trials_list:
        attrs = {
            "classname": t.adapter,
            "name": t.trial_id,
        }
        testcase = SubElement(testsuite, "testcase", attrs)

        props = SubElement(testcase, "properties")
        SubElement(props, "property", {"name": "task_id", "value": t.task_id})
        SubElement(props, "property", {"name": "seed", "value": str(t.seed)})
        if t.started_at:
            SubElement(props, "property", {"name": "started_at", "value": _iso(t.started_at) or ""})
        if t.finished_at:
            SubElement(props, "property", {"name": "finished_at", "value": _iso(t.finished_at) or ""})

        if not t.ok:
            errors += 1
            SubElement(testcase, "error", {"message": t.error or "trial_error"})
            continue

        passed = True
        if t.graders:
            passed = all(g.passed for g in t.graders)

        if not passed:
            failures += 1
            failure_el = SubElement(testcase, "failure", {"message": "grader_failed"})
            failure_el.text = "\n".join(
                [f"{g.name}: passed={g.passed} score={g.score} details={g.details}" for g in t.graders]
            )

    testsuite.set("failures", str(failures))
    testsuite.set("errors", str(errors))

    testsuites = Element("testsuites")
    testsuites.append(testsuite)
    return tostring(testsuites, encoding="unicode")

