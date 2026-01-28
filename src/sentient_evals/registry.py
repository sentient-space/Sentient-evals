from __future__ import annotations

import importlib
from typing import Any

from .adapters import ExampleInstalledAdapter, WorkflowStubAdapter
from .graders import (
    BudgetGrader,
    ExactMatchGrader,
    HumanReviewGrader,
    HumanReviewSpec,
    LLMJudgeConfig,
    LLMJudgeGrader,
    MultiJudgeAggregation,
    MultiLLMJudgeGrader,
    PairwiseJudgeConfig,
    PairwiseJudgeGrader,
    StateCheckGrader,
    StaticAnalysisGrader,
    StaticAnalysisSpec,
    ToolUsageGrader,
    ToolUsageRule,
    VerifierScriptGrader,
    VerifierScriptSpec,
)


def _load_object(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError("import_path must be like 'module:object'")
    mod, name = import_path.split(":", 1)
    m = importlib.import_module(mod)
    return getattr(m, name)


def build_adapter(
    *, adapter_type: str, import_path: str | None = None, kwargs: dict[str, Any] | None = None
) -> Any:
    kwargs = kwargs or {}
    if adapter_type == "workflow_stub":
        return WorkflowStubAdapter()
    if adapter_type == "example_installed":
        return ExampleInstalledAdapter()
    if adapter_type == "import":
        if not import_path:
            raise ValueError("adapter type=import requires import_path")
        obj = _load_object(import_path)
        if isinstance(obj, type):
            return obj(**kwargs)
        if callable(obj):
            return obj(**kwargs)
        raise ValueError("imported adapter must be a class or callable")
    raise ValueError(f"Unknown adapter type: {adapter_type}")


def build_grader(spec: dict[str, Any]) -> Any:
    t = str(spec.get("type", "")).strip()
    cfg = spec.get("config") or {}
    if t == "exact_match":
        return ExactMatchGrader(**cfg)
    if t == "state_check":
        return StateCheckGrader(**cfg)
    if t == "tool_usage":
        rules = [ToolUsageRule(**r) for r in cfg.get("rules", [])] if isinstance(cfg.get("rules"), list) else []
        return ToolUsageGrader(
            required_tools=cfg.get("required_tools", ()),
            forbidden_tools=cfg.get("forbidden_tools", ()),
            rules=rules,
            max_turns=cfg.get("max_turns"),
            max_tool_calls=cfg.get("max_tool_calls"),
        )
    if t == "static_analysis":
        checks = [StaticAnalysisSpec(**c) for c in cfg.get("checks", [])]
        return StaticAnalysisGrader(checks=checks)
    if t == "verifier_script":
        spec_obj = VerifierScriptSpec(**cfg) if cfg else VerifierScriptSpec()
        return VerifierScriptGrader(spec=spec_obj)
    if t == "budget":
        return BudgetGrader(**cfg)
    if t == "human_review":
        spec_obj = HumanReviewSpec(**cfg) if cfg else HumanReviewSpec()
        return HumanReviewGrader(spec=spec_obj)
    if t == "llm_judge":
        judge_cfg = LLMJudgeConfig(**cfg["judge"]) if "judge" in cfg else None
        return LLMJudgeGrader(
            rubric=cfg.get("rubric", "Decide whether the model answer satisfies the task."),
            reference=cfg.get("reference"),
            answer_field=cfg.get("answer_field", "answer"),
            config=judge_cfg,
        )
    if t == "multi_llm_judge":
        agg = MultiJudgeAggregation(cfg.get("aggregation", MultiJudgeAggregation.majority))
        judges = [build_grader(j) for j in cfg.get("judges", [])]
        llm_judges = [j for j in judges if isinstance(j, LLMJudgeGrader)]
        return MultiLLMJudgeGrader(judges=llm_judges, aggregation=agg)
    if t == "pairwise_judge":
        judge_cfg = PairwiseJudgeConfig(**cfg["judge"]) if "judge" in cfg else None
        return PairwiseJudgeGrader(
            rubric=cfg.get("rubric", "Compare the two responses and pick the better one."),
            config=judge_cfg,
            candidate_key=cfg.get("candidate_key", "candidate"),
            baseline_key=cfg.get("baseline_key", "baseline"),
        )
    raise ValueError(f"Unknown grader type: {t}")

