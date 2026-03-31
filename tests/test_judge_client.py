from __future__ import annotations

import asyncio

from sentient_evals.judges import DirectJudgeClient


class _StubJudgeClient(DirectJudgeClient):
    def __init__(self) -> None:
        super().__init__(api_key="test-key")
        object.__setattr__(self, "calls", [])

    async def _post_json(self, url: str, *, headers: dict[str, str], body: dict):
        self.calls.append((url, headers, body))
        if "anthropic.com" in url:
            return {"content": [{"type": "text", "text": '{"verdict":"pass","score":1.0,"reason":"ok"}'}]}
        if "openrouter.ai" in url:
            return {"choices": [{"message": {"content": '{"verdict":"pass","score":1.0,"reason":"ok"}'}}]}
        return {"choices": [{"message": {"content": '{"verdict":"pass","score":1.0,"reason":"ok"}'}}]}


def test_direct_judge_client_routes_prefixed_anthropic_model():
    client = _StubJudgeClient()
    response = asyncio.run(
        client.score(
            model="anthropic/claude-sonnet-4-5",
            prompt="judge this",
            max_tokens=128,
            temperature=0.0,
        )
    )
    assert response.content == '{"verdict":"pass","score":1.0,"reason":"ok"}'
    assert client.calls
    url, headers, body = client.calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "test-key"
    assert body["model"] == "claude-sonnet-4-5"


def test_direct_judge_client_preserves_openrouter_model_id():
    client = _StubJudgeClient()
    response = asyncio.run(
        client.score(
            model="openrouter/openai/gpt-4o-mini",
            prompt="judge this",
            max_tokens=64,
            temperature=0.1,
        )
    )
    assert response.content == '{"verdict":"pass","score":1.0,"reason":"ok"}'
    assert client.calls
    url, headers, body = client.calls[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "openai/gpt-4o-mini"
