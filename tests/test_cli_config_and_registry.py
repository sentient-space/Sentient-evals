import json
from pathlib import Path

from sentient_evals.config_files import load_run_spec
from sentient_evals.registry import build_grader


def test_load_run_spec_toml(tmp_path: Path):
    p = tmp_path / "eval.toml"
    p.write_text(
        '\n'.join(
            [
                "[adapter]",
                'type = "workflow_stub"',
                "",
                "[[graders]]",
                'type = "verifier_script"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    spec = load_run_spec(p)
    assert spec.adapter.type == "workflow_stub"
    assert spec.graders[0]["type"] == "verifier_script"


def test_load_run_spec_json(tmp_path: Path):
    p = tmp_path / "eval.json"
    p.write_text(
        json.dumps({"adapter": {"type": "workflow_stub"}, "graders": [{"type": "budget", "config": {"max_tokens": 1}}]}),
        encoding="utf-8",
    )
    spec = load_run_spec(p)
    assert spec.adapter.type == "workflow_stub"
    assert spec.graders[0]["type"] == "budget"


def test_build_grader_budget():
    g = build_grader({"type": "budget", "config": {"max_tokens": 1}})
    assert getattr(g, "name") == "budget"
