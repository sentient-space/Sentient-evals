from __future__ import annotations

import importlib.util
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Awaitable, Callable, get_args, get_origin

from .adapters import AgentAdapter
from .artifacts import TrialArtifacts
from .env import ToolExecutor
from .models import Outcome, Task, TranscriptEvent


@dataclass(frozen=True)
class AgentFileRef:
    path: Path
    attr: str


def parse_agent_file_ref(value: str) -> AgentFileRef:
    if ":" not in value:
        raise ValueError("agent ref must be like './path/to/file.py:agent_fn'")
    p, attr = value.split(":", 1)
    path = Path(p).expanduser().resolve()
    if path.suffix != ".py":
        raise ValueError("agent file must be a .py file")
    if not attr:
        raise ValueError("agent ref must include an attribute after ':'")
    return AgentFileRef(path=path, attr=attr)


def _load_module_from_file(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(str(path))
    module_name = f"sentient_evals_user_agent_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _as_async(fn: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    if inspect.iscoroutinefunction(fn):
        return fn  # type: ignore[return-value]

    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return _wrapped


def _annotation_is_str(annotation: Any) -> bool:
    if annotation is str:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    if origin is list or origin is dict:
        return False
    return any(a is str for a in get_args(annotation))


def _best_effort_prompt(task_input: dict[str, Any]) -> str:
    for key in ("input", "prompt", "query", "question", "text"):
        v = task_input.get(key)
        if isinstance(v, str) and v.strip():
            return v
    if len(task_input) == 1:
        only = next(iter(task_input.values()))
        if isinstance(only, str) and only.strip():
            return only
    return json.dumps(task_input, ensure_ascii=False)


def _call_kwargs_for(
    fn: Callable[..., Any],
    *,
    task: Task,
    instruction: str | None,
    env: ToolExecutor,
    seed: int,
    artifacts: TrialArtifacts,
) -> dict[str, Any]:
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    prompt = _best_effort_prompt(task.input)
    for name, param in sig.parameters.items():
        if name in {"inputs", "payload"}:
            kwargs[name] = task.input
        elif name == "input":
            kwargs[name] = prompt if _annotation_is_str(param.annotation) else task.input
        elif name in {"instruction"}:
            kwargs[name] = instruction or prompt
        elif name in {"task"}:
            kwargs[name] = task
        elif name in {"tools", "env"}:
            kwargs[name] = env
        elif name in {"seed"}:
            kwargs[name] = seed
        elif name in {"artifacts"}:
            kwargs[name] = artifacts
        elif name in {"context"}:
            ctx = dict(task.metadata or {})
            ctx.setdefault("stream", False)
            kwargs[name] = ctx
    return kwargs


class CallableAgentAdapter:
    def __init__(self, fn: Callable[..., Any], *, name: str | None = None) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "callable_agent")

    async def run(
        self,
        task: Task,
        *,
        instruction: str | None = None,
        seed: int,
        env: ToolExecutor,
        artifacts: TrialArtifacts | None = None,
    ) -> tuple[list[TranscriptEvent], Outcome]:
        if artifacts is None:
            # Backwards-compatible default for direct calls/tests that don't provide artifacts.
            # Runner paths always pass a trial-scoped artifacts directory.
            artifacts = TrialArtifacts(Path.cwd() / ".sentient-evals-artifacts")

        async_fn = _as_async(self._fn)
        kwargs = _call_kwargs_for(
            self._fn, task=task, instruction=instruction, env=env, seed=seed, artifacts=artifacts
        )

        transcript: list[TranscriptEvent] = [
            TranscriptEvent(kind="message", role="user", content=str(kwargs.get("input", task.input)))
        ]
        result = await async_fn(**kwargs)

        if isinstance(result, tuple) and len(result) == 2:
            tr, out = result
            if isinstance(out, Outcome) and isinstance(tr, list):
                return tr, out

        if inspect.isasyncgen(result) or hasattr(result, "__aiter__"):
            parts: list[str] = []
            async for chunk in result:  # type: ignore[misc]
                parts.append(str(chunk))
            text = "".join(parts)
            transcript.append(TranscriptEvent(kind="message", role="assistant", content=text))
            return transcript, Outcome(summary="ok", data={"output": text, "answer": text})

        if isinstance(result, Outcome):
            transcript.append(TranscriptEvent(kind="message", role="assistant", content=result.summary or ""))
            return transcript, result

        if isinstance(result, dict):
            transcript.append(TranscriptEvent(kind="message", role="assistant", content=str(result)))
            return transcript, Outcome(summary="ok", data=result)

        if isinstance(result, str):
            transcript.append(TranscriptEvent(kind="message", role="assistant", content=result))
            return transcript, Outcome(summary="ok", data={"output": result, "answer": result})

        transcript.append(TranscriptEvent(kind="message", role="assistant", content=str(result)))
        return transcript, Outcome(summary="ok", data={"output": str(result), "answer": str(result)})


def load_agent_adapter_from_file(ref: AgentFileRef) -> AgentAdapter:
    mod = _load_module_from_file(ref.path)
    obj = getattr(mod, ref.attr)

    if hasattr(obj, "run") and callable(getattr(obj, "run")):
        return obj  # type: ignore[return-value]

    if callable(obj):
        sig = inspect.signature(obj)
        if len(sig.parameters) == 0:
            built = obj()
            if hasattr(built, "run") and callable(getattr(built, "run")):
                return built  # type: ignore[return-value]
        return CallableAgentAdapter(obj, name=ref.attr)

    raise ValueError(f"Unsupported agent attribute: {ref.attr}")
