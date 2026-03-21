import json
from pathlib import Path

from sentient_evals.adapters.installed.parsers.claude_code import parse_claude_code_session
from sentient_evals.adapters.installed.parsers.codex import parse_codex_exec_output, parse_codex_session
from sentient_evals.adapters.installed.parsers.gemini_cli import parse_gemini_trajectory
from sentient_evals.adapters.installed.parsers.mini_swe_agent import parse_mini_swe_agent_trajectory
from sentient_evals.adapters.installed.parsers.openhands import parse_openhands_session
from sentient_evals.adapters.installed.parsers.swe_agent import parse_swe_agent_traj


def test_parse_swe_agent_traj_emits_tool_calls(tmp_path: Path):
    traj = {
        "info": {"input_tokens": 10, "output_tokens": 5, "total_cost": 0.01},
        "trajectory": [
            {"response": "ok", "thought": "think", "action": "echo hi", "observation": "hi"},
        ],
    }
    p = tmp_path / "swe.traj"
    p.write_text(json.dumps(traj), encoding="utf-8")
    parsed = parse_swe_agent_traj(p, instruction="do it", model_name="x")
    assert parsed is not None
    assert any(e.kind == "tool_call" for e in parsed.events)
    assert parsed.metrics.get("prompt_tokens") == 10.0


def test_parse_mini_swe_agent_emits_messages(tmp_path: Path):
    data = {
        "info": {"model_stats": {"instance_cost": 0.02}},
        "messages": [
            {"role": "assistant", "content": "hello", "extra": {"response": {"usage": {"prompt_tokens": 2, "completion_tokens": 3}}}},
        ],
    }
    p = tmp_path / "mini.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    parsed = parse_mini_swe_agent_trajectory(p, instruction="do", model_name="m")
    assert parsed is not None
    assert any(e.kind == "message" and e.role == "assistant" for e in parsed.events)
    assert parsed.metrics.get("completion_tokens") == 3.0


def test_parse_openhands_tool_call_metadata(tmp_path: Path):
    sess = tmp_path / "sessions" / "s1" / "events"
    sess.mkdir(parents=True, exist_ok=True)
    (sess / "0.json").write_text(json.dumps({"source": "agent", "tool_call_metadata": {"tool_call_id": "t1", "function_name": "exec", "model_response": {"choices": [{"message": {"tool_calls": [{"function": {"arguments": "{\"cmd\":\"ls\"}"}}]}}]}}}), encoding="utf-8")
    (sess / "1.json").write_text(json.dumps({"source": "agent", "cause": "x", "observation": True, "content": "ok"}), encoding="utf-8")
    parsed = parse_openhands_session(tmp_path, instruction="do")
    assert parsed is not None
    tool_calls = [e for e in parsed.events if e.kind == "tool_call" and e.tool_call]
    assert tool_calls
    assert tool_calls[0].tool_call.id == "t1"
    assert tool_calls[0].observation == "ok"



def test_parse_codex_jsonl(tmp_path: Path):
    sessions = tmp_path / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    p_old = sessions / "old.jsonl"
    p_old.write_text(
        '\n'.join(
            [
                '{"type":"message","payload":{"role":"assistant","text":"hello"}}',
                '{"type":"tool_call","payload":{"call_id":"c1","tool_name":"exec","arguments":{"cmd":"ls"},"output":"x"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    p_new = sessions / "new.jsonl"
    p_new.write_text(
        '\n'.join(
            [
                '{"type":"message","payload":{"role":"assistant","text":"new"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    import os, time
    os.utime(p_old, (time.time() - 10, time.time() - 10))
    os.utime(p_new, (time.time(), time.time()))

    parsed = parse_codex_session(tmp_path, instruction="do")
    assert parsed is not None
    # Must select newest session file (which does NOT contain tool_call).
    assert any(e.kind == "message" and e.role == "assistant" and (e.content or "") == "new" for e in parsed.events)


def test_parse_codex_modern_session_jsonl(tmp_path: Path):
    sessions = tmp_path / "sessions" / "2026" / "03" / "21"
    sessions.mkdir(parents=True, exist_ok=True)
    p = sessions / "rollout.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-03-21T00:00:00Z", "type": "session_meta", "payload": {"cli_version": "0.116.0"}}),
                json.dumps(
                    {
                        "timestamp": "2026-03-21T00:00:01Z",
                        "type": "response_item",
                        "payload": {"type": "reasoning", "summary": [{"text": "Inspect the environment first."}]},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-21T00:00:02Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "phase": "commentary",
                            "content": [{"type": "output_text", "text": "Checking Composio setup."}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-21T00:00:03Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "call_id": "call_1",
                            "arguments": json.dumps({"cmd": "which composio"}),
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-21T00:00:04Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "Command: which composio\nOutput:\n/root/.composio/composio\n",
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-21T00:00:05Z",
                        "type": "response_item",
                        "payload": {
                            "type": "web_search_call",
                            "status": "completed",
                            "action": {"type": "search", "query": "composio gmail send email"},
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-21T00:00:06Z",
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 100,
                                    "cached_input_tokens": 40,
                                    "output_tokens": 12,
                                }
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-21T00:00:07Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "phase": "final_answer",
                            "content": [{"type": "output_text", "text": "Email sent."}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_codex_session(tmp_path, instruction="send the email")
    assert parsed is not None
    assert parsed.metrics["prompt_tokens"] == 100.0
    assert parsed.metrics["cached_tokens"] == 40.0
    assert parsed.metrics["completion_tokens"] == 12.0
    assert any(e.kind == "message" and e.role == "assistant" and (e.content or "") == "Checking Composio setup." for e in parsed.events)
    tool_calls = [e for e in parsed.events if e.kind == "tool_call" and e.tool_call]
    assert any(e.tool_call and e.tool_call.id == "call_1" and e.tool_call.name == "exec_command" for e in tool_calls)
    exec_event = next(e for e in tool_calls if e.tool_call and e.tool_call.id == "call_1")
    assert exec_event.observation == "Command: which composio\nOutput:\n/root/.composio/composio\n"
    assert any(e.reasoning_content == "Inspect the environment first." for e in parsed.events)
    assert any(e.tool_call and e.tool_call.name == "search" for e in tool_calls)
    assert any(e.kind == "message" and e.role == "assistant" and (e.content or "") == "Email sent." for e in parsed.events)


def test_parse_codex_exec_output_stream(tmp_path: Path):
    del tmp_path
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "t1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "item_0", "type": "agent_message", "text": "Checking Composio setup."},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_1",
                        "type": "command_execution",
                        "command": "/bin/bash -lc 'which composio'",
                        "aggregated_output": "/root/.composio/composio\n",
                        "exit_code": 0,
                        "status": "completed",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "item_2", "type": "agent_message", "text": "Email sent."},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 50, "cached_input_tokens": 10, "output_tokens": 5},
                }
            ),
        ]
    )
    parsed = parse_codex_exec_output(stdout, instruction="send the email")
    assert parsed is not None
    assert parsed.metrics["prompt_tokens"] == 50.0
    assert parsed.metrics["cached_tokens"] == 10.0
    assert parsed.metrics["completion_tokens"] == 5.0
    assert any(e.kind == "message" and e.role == "assistant" and (e.content or "") == "Checking Composio setup." for e in parsed.events)
    tool_calls = [e for e in parsed.events if e.kind == "tool_call" and e.tool_call]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call.name == "exec_command"
    assert tool_calls[0].observation == {
        "stdout": "/root/.composio/composio\n",
        "exit_code": 0,
        "status": "completed",
        "command": "/bin/bash -lc 'which composio'",
    }


def test_parse_claude_stream_json_basic(tmp_path: Path):
    sessions = tmp_path / "sessions" / "projects" / "p"
    sessions.mkdir(parents=True, exist_ok=True)
    p = sessions / "s.jsonl"
    p.write_text(
        '\n'.join(
            [
                json.dumps({"type": "assistant", "timestamp": "2026-01-30T00:00:00Z", "message": {"role": "assistant", "content": "hello", "usage": {"input_tokens": 1, "output_tokens": 2}}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_claude_code_session(tmp_path, instruction="do")
    assert parsed is not None
    assert any(e.kind == "message" and e.role == "assistant" for e in parsed.events)


def test_parse_gemini_trajectory(tmp_path: Path):
    p = tmp_path / "gemini-cli.trajectory.json"
    p.write_text(
        json.dumps(
            {
                "sessionId": "s",
                "messages": [
                    {"type": "user", "content": "hi"},
                    {
                        "type": "gemini",
                        "content": "ok",
                        "thoughts": [{"subject": "plan", "description": "do"}],
                        "tokens": {"input": 1, "output": 2, "cached": 0, "thoughts": 1, "tool": 0},
                        "toolCalls": [
                            {
                                "id": "t1",
                                "name": "exec",
                                "args": {"cmd": "ls"},
                                "result": [{"functionResponse": {"response": {"output": "x"}}}],
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    parsed = parse_gemini_trajectory(tmp_path, instruction="do")
    assert parsed is not None
    assert any(e.kind == "tool_call" and e.tool_call and e.tool_call.id == "t1" for e in parsed.events)


def test_final_metrics_populated_from_transcript(tmp_path: Path):
    from sentient_evals.atif.converters import transcript_to_trajectory
    from sentient_evals.models import TranscriptEvent

    t = [
        TranscriptEvent(kind="message", role="user", content="x"),
        TranscriptEvent(kind="message", role="assistant", content="y", metrics={"prompt_tokens": 2, "completion_tokens": 3}),
    ]
    traj = transcript_to_trajectory(t, session_id="s", agent_name="a", agent_version="v")
    assert traj.final_metrics is not None
    assert traj.final_metrics.total_prompt_tokens == 2
    assert traj.final_metrics.total_completion_tokens == 3
