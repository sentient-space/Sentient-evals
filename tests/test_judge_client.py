from __future__ import annotations

import asyncio

from sentient_evals.judges import DirectJudgeClient
from sentient_evals.judges.parsing import parse_judge_response


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


def test_judge_parser_accepts_markdown_fenced_json():
    parsed = parse_judge_response(
        """```json
{
  "verdict": "unknown",
  "score": 0.5,
  "reason": "Cannot verify."
}
```"""
    )

    assert parsed.verdict == "unknown"
    assert parsed.score == 0.5
    assert parsed.normalized_score == 0.5
    assert parsed.reason == "Cannot verify."


def test_judge_parser_supports_scaled_scores_and_reasoning_alias():
    parsed = parse_judge_response(
        'The answer is:\n{"verdict":"pass","score":4,"reasoning":"Good."}',
        score_scale="1-5",
    )

    assert parsed.verdict == "pass"
    assert parsed.score == 4
    assert parsed.normalized_score == 0.8
    assert parsed.reason == "Good."
