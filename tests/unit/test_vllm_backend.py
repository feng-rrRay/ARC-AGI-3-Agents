"""Unit tests for VLLMBackend (no server, no GPU — the OpenAI client is faked).

Covers the contract that lets the open-weight backend slot into the harness
unchanged:
  * text-only models drop images; vision models emit image_url parts;
  * the duck-typed response is parsed by the REAL harness extractors
    (extract_function_calls / parse_action_response / serialize_response);
  * tool results are linked to the right tool_call_id by index;
  * OpenAI usage is normalised into the canonical {prompt, output, ...} shape.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from arcengine import GameAction
from PIL import Image

from agents.templates.continual_harness.helpers import parse_action_response
from agents.templates.continual_harness.tools import extract_function_calls
from agents.templates.continual_harness.trace import serialize_response
from agents.templates.utils.vlm_backend import VLLMBackend, _build_vllm_command

pytestmark = pytest.mark.unit


# --- fake OpenAI SDK objects --------------------------------------------------


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    reasoning: str | None = None,
    with_usage: bool = True,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        reasoning_content=reasoning,
    )
    usage = None
    if with_usage:
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=8),
            prompt_tokens_details=SimpleNamespace(cached_tokens=10),
        )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
        usage=usage,
    )


class _FakeCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))


def _backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vision: bool = False,
    model: str = "gpt-oss-120b",
    response: Any = None,
) -> VLLMBackend:
    monkeypatch.setenv("VLLM_AUTO_START", "false")  # never launch a real server
    monkeypatch.setenv("VLLM_SUPPORTS_VISION", "true" if vision else "false")
    backend = VLLMBackend(model)
    if response is not None:
        backend.client = _FakeClient(response)  # type: ignore[assignment]
    return backend


# --- tests --------------------------------------------------------------------


def test_text_only_drops_images(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(monkeypatch, vision=False)
    msg = backend._user_message("hello", [Image.new("RGBA", (8, 8))])
    assert msg == {"role": "user", "content": "hello"}


def test_vision_emits_image_url(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(monkeypatch, vision=True, model="qwen3.6-27b")
    msg = backend._user_message("hi", [Image.new("RGBA", (8, 8))])
    assert isinstance(msg["content"], list)
    kinds = [part["type"] for part in msg["content"]]
    assert kinds == ["text", "image_url"]
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_response_parsed_by_real_extractors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _completion(
        content="picking action 1",
        reasoning="the door is to the left",
        tool_calls=[_tool_call("call_0", "ACTION1", json.dumps({"reasoning": "go"}))],
    )
    backend = _backend(monkeypatch, response=response)
    backend.set_tools(
        [{"name": "ACTION1", "description": "move", "parameters": {"type": "object"}}]
    )

    resp = backend.get_query_contents([backend.build_user_turn("prompt", [])])

    fcs = extract_function_calls(resp)
    assert [(fc.name, fc.args) for fc in fcs] == [("ACTION1", {"reasoning": "go"})]

    serialized = serialize_response(resp)
    assert serialized["function_calls"] == [
        {"name": "ACTION1", "args": {"reasoning": "go"}}
    ]
    assert serialized["finish_reason"] == "stop"
    assert "the door is to the left" in serialized["text"]

    action = parse_action_response(resp, [GameAction.ACTION1])
    assert action is GameAction.ACTION1
    assert action.reasoning == "go"


def test_tool_call_id_correlation(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _completion(
        tool_calls=[_tool_call("abc-123", "run_skill", json.dumps({"id": "s1"}))],
    )
    backend = _backend(monkeypatch, response=response)
    backend.set_tools(
        [{"name": "run_skill", "description": "", "parameters": {"type": "object"}}]
    )

    resp = backend.get_query_contents([backend.build_user_turn("p", [])])
    contents = [
        backend.build_user_turn("p", []),
        backend.model_turn(resp),
        backend.build_tool_results_turn([("run_skill", {"ok": True})]),
    ]
    messages = backend._to_openai_messages(contents)

    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "abc-123"
    assert json.loads(tool_messages[0]["content"]) == {"ok": True}

    assistant = [m for m in messages if m["role"] == "assistant"]
    assert assistant and assistant[0]["tool_calls"][0]["id"] == "abc-123"


def test_usage_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(monkeypatch, response=_completion(content="x"))
    resp = backend.get_query_contents([backend.build_user_turn("p", [])])
    usage = backend.extract_usage(resp)
    assert usage == {
        "prompt": 100,
        "output": 20,
        "total": 120,
        "thoughts": 8,
        "cached": 10,
        "tool_use": None,
    }


def test_no_tools_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(monkeypatch, response=_completion(content="hello world"))
    # No tools set → text-style return (mirrors GeminiBackend behaviour).
    assert backend.get_text_query("ping") == "hello world"


def test_system_instruction_is_leading_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _completion(content="ok")
    backend = _backend(monkeypatch, response=response)
    backend.set_system_instruction("you are a player")
    backend.get_query_contents([backend.build_user_turn("p", [])])
    sent = backend.client.chat.completions.calls[-1]  # type: ignore[attr-defined]
    assert sent["messages"][0] == {"role": "system", "content": "you are a player"}


def test_context_window_from_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(monkeypatch, model="gpt-oss-120b")
    assert backend.context_window_tokens() == 128_000


# --- server command builder (speculative / MTP config) -----------------------


def _build_cmd(**overrides: Any) -> list[str]:
    base: dict[str, Any] = dict(
        model_path="/m",
        served_model_name="x",
        host="127.0.0.1",
        port=8000,
        tool_parser="glm47",
        max_model_len=128_000,
        reasoning_parser="glm45",
        tensor_parallel_size=1,
        kv_cache_dtype="auto",
        enforce_eager=False,
        gpu_memory_utilization=0.9,
        extra_args=(),
    )
    base.update(overrides)
    return _build_vllm_command(**base)


def _flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_speculative_config_glm() -> None:
    cmd = _build_cmd(speculative_method="mtp", num_speculative_tokens=1)
    assert json.loads(_flag_value(cmd, "--speculative-config")) == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }


def test_speculative_config_qwen() -> None:
    cmd = _build_cmd(speculative_method="qwen3_next_mtp", num_speculative_tokens=2)
    assert json.loads(_flag_value(cmd, "--speculative-config")) == {
        "method": "qwen3_next_mtp",
        "num_speculative_tokens": 2,
    }


def test_speculative_config_absent_when_unset() -> None:
    assert "--speculative-config" not in _build_cmd()
