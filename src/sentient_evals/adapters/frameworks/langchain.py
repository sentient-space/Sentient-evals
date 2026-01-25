from __future__ import annotations

from typing import Any, Callable

from sentient_evals.adapters.frameworks.common import ToolExecutorRecorder, build_instruction_payload
from sentient_evals.env import ToolExecutor
from sentient_evals.models import Outcome, Task, TranscriptEvent


class LangChainAdapter:
    name = "langchain"

    def __init__(
        self,
        *,
        agent_factory: Callable[[list[Any]], Any],
        input_key: str = "input",
    ) -> None:
        self.agent_factory = agent_factory
        self.input_key = input_key

    async def run(
        self, task: Task, *, seed: int, env: ToolExecutor
    ) -> tuple[list[TranscriptEvent], Outcome]:
        try:
            from langchain_core.tools import Tool  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Install sentient-evals[langchain] to use LangChainAdapter"
            ) from exc

        instruction = build_instruction_payload(task.input)
        transcript: list[TranscriptEvent] = [
            TranscriptEvent(kind="message", role="user", content=instruction)
        ]
        recorder = ToolExecutorRecorder(env, transcript)

        async def exec_tool(cmd: str) -> str:
            res = await recorder.exec(cmd)
            return res.stdout

        async def read_tool(path: str) -> str:
            return await recorder.read_file(path)

        async def write_tool(path: str, content: str) -> str:
            await recorder.write_file(path, content)
            return "ok"

        async def list_tool(path: str) -> list[str]:
            return await recorder.list_dir(path)

        async def call_tool(tool_name: str, args: dict[str, Any]) -> Any:
            return await recorder.call(tool_name, args)

        tools = [
            Tool(name="exec", description="Execute a shell command.", coroutine=exec_tool, func=None),
            Tool(name="read_file", description="Read a file path.", coroutine=read_tool, func=None),
            Tool(name="write_file", description="Write file content.", coroutine=write_tool, func=None),
            Tool(name="list_dir", description="List a directory.", coroutine=list_tool, func=None),
            Tool(name="call", description="Call a named tool.", coroutine=call_tool, func=None),
        ]

        agent = self.agent_factory(tools)
        if hasattr(agent, "ainvoke"):
            result = await agent.ainvoke({self.input_key: instruction})
        elif callable(agent):
            result = await agent({self.input_key: instruction})
        else:
            raise RuntimeError("agent_factory must return a Runnable with ainvoke")

        transcript.append(
            TranscriptEvent(kind="message", role="assistant", content=str(result))
        )
        return transcript, Outcome(summary="ok", data={"answer": result})
