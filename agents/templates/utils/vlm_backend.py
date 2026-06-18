from __future__ import annotations

import atexit
import base64
import io
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence, TypeAlias, cast
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ImageInput: TypeAlias = Image.Image | np.ndarray[Any, Any]


@dataclass(frozen=True, slots=True)
class VLMUsagePricing:
    input_per_m: float
    output_per_m: float
    cached_per_m: float
    tier: str


def usage_token_count(usage: dict[str, Any] | None, key: str) -> int:
    if usage is None:
        return 0
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _gemini_usage_pricing(
    model_name: str, prompt_tokens: int
) -> VLMUsagePricing | None:
    model = model_name.lower()
    if "gemini-3.1-pro-preview" in model:
        if prompt_tokens > 200_000:
            return VLMUsagePricing(
                input_per_m=4.00,
                output_per_m=18.00,
                cached_per_m=0.40,
                tier="gt_200k",
            )
        return VLMUsagePricing(
            input_per_m=2.00,
            output_per_m=12.00,
            cached_per_m=0.20,
            tier="le_200k",
        )
    if "gemini-3.5-flash" in model:
        return VLMUsagePricing(
            input_per_m=1.50,
            output_per_m=9.00,
            cached_per_m=0.15,
            tier="flat",
        )
    if "gemini-3-flash-preview" in model:
        return VLMUsagePricing(
            input_per_m=0.50,
            output_per_m=3.00,
            cached_per_m=0.05,
            tier="flat",
        )
    return None


def estimate_vlm_usage_cost(
    model_name: str,
    usage: dict[str, Any] | None,
    *,
    cumulative_usd_before: float = 0.0,
) -> dict[str, Any] | None:
    if usage is None:
        return None

    prompt_tokens = usage_token_count(usage, "prompt")
    output_tokens = usage_token_count(usage, "output")
    thoughts_tokens = usage_token_count(usage, "thoughts")
    cached_tokens = min(usage_token_count(usage, "cached"), prompt_tokens)

    pricing = _gemini_usage_pricing(model_name, prompt_tokens)
    if pricing is None:
        return None

    uncached_input_tokens = max(0, prompt_tokens - cached_tokens)
    billable_output_tokens = output_tokens + thoughts_tokens
    current_cost = (
        (uncached_input_tokens * pricing.input_per_m)
        + (cached_tokens * pricing.cached_per_m)
        + (billable_output_tokens * pricing.output_per_m)
    ) / 1_000_000
    return {
        "model": model_name,
        "tier": pricing.tier,
        "current_usd": current_cost,
        "cumulative_usd": cumulative_usd_before + current_cost,
        "prompt_tokens": prompt_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "thoughts_tokens": thoughts_tokens,
        "billable_output_tokens": billable_output_tokens,
        "rates_per_m": {
            "input": pricing.input_per_m,
            "output": pricing.output_per_m,
            "cached": pricing.cached_per_m,
        },
    }


class VLMBackend(ABC):
    """Abstract base class for VLM backends."""

    @abstractmethod
    def get_query(
        self,
        img: ImageInput | Sequence[ImageInput],
        text: str,
        module_name: str = "Unknown",
    ) -> Any:
        """Send a prompt + image(s) and return the provider response (or text if no tools)."""
        raise NotImplementedError

    @abstractmethod
    def get_text_query(self, text: str, module_name: str = "Unknown") -> Any:
        """Send a text-only prompt."""
        raise NotImplementedError

    def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
        """Default: ignore. Backends that support function calling override this."""
        return None

    def set_system_instruction(self, text: str | None) -> None:
        """Swap the system instruction in place.

        Default impl assigns to `self.system_instruction`; backends that need
        to rebuild a client object override. The Gemini backend re-reads
        `system_instruction` on every call via `_config()`, so the default is
        sufficient.
        """
        # Subclasses are expected to declare `system_instruction: str | None`
        # in their __init__; this base just provides the swap semantics.
        self.system_instruction = text

    def extract_usage(self, response: Any) -> dict[str, int | None] | None:
        """Return canonical token usage {prompt, output, total, thoughts, cached, tool_use}.

        Default: provider doesn't report usage → return None. Backends with a
        usage payload override this and normalise into the canonical shape so
        agent/trace code never sees provider-specific field names.
        """
        return None

    # --- Multi-turn conversation API ------------------------------------
    # Used by the orchestrator's conversation-until-action loop. Provider
    # "turn"/"content" objects are opaque to callers — build them with these
    # helpers and feed the accumulated list back to `get_query_contents`.

    def build_user_turn(self, text: str, images: Sequence[ImageInput]) -> Any:
        """Build one user turn (prompt text + image parts)."""
        raise NotImplementedError

    def build_tool_results_turn(self, pairs: Sequence[tuple[str, dict[str, Any]]]) -> Any:
        """Build one user turn carrying function-response parts (name -> response dict)."""
        raise NotImplementedError

    def model_turn(self, response: Any) -> Any:
        """Extract the model turn from a response, to append back to the conversation."""
        raise NotImplementedError

    def get_query_contents(self, contents: list[Any], module_name: str = "Unknown") -> Any:
        """Send a full multi-turn conversation (list of turns) and return the response."""
        raise NotImplementedError

    def count_input_tokens(
        self, contents: list[Any], module_name: str = "Unknown"
    ) -> int:
        """Return an estimated input-token count for a generate request."""
        return _estimate_input_tokens(contents)

    def context_window_tokens(self) -> int | None:
        """Return the model input context window in tokens, if known."""
        return None


def _estimate_input_tokens(value: Any) -> int:
    """Conservative local fallback for request sizing when provider counting fails."""
    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, (len(value) + 3) // 4)
    if isinstance(value, Image.Image):
        return 258
    if isinstance(value, np.ndarray):
        return 258
    if isinstance(value, bytes):
        return max(1, len(value) // 3)
    if isinstance(value, dict):
        return sum(_estimate_input_tokens(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_estimate_input_tokens(v) for v in value)

    total = 0
    parts = getattr(value, "parts", None)
    if parts is not None:
        total += _estimate_input_tokens(parts)
    text = getattr(value, "text", None)
    if text:
        total += _estimate_input_tokens(str(text))
    data = getattr(value, "data", None)
    if data is not None:
        total += _estimate_input_tokens(data)
    if total:
        return total
    return max(1, (len(str(value)) + 3) // 4)


class _PlaceholderBackend(VLMBackend):
    """Base for not-yet-implemented backends. Raises on use; accepts set_tools as a no-op."""

    provider_name: ClassVar[str] = "Provider"

    def __init__(
        self,
        model_name: str,
        tools: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
        **_: Any,
    ) -> None:
        self.model_name = model_name
        self.tools = tools or []
        self.system_instruction = system_instruction

    def get_query(
        self,
        img: ImageInput | Sequence[ImageInput],
        text: str,
        module_name: str = "Unknown",
    ) -> Any:
        raise NotImplementedError(
            f"{self.provider_name} backend is a placeholder in this repo."
        )

    def get_text_query(self, text: str, module_name: str = "Unknown") -> Any:
        raise NotImplementedError(
            f"{self.provider_name} backend is a placeholder in this repo."
        )

    def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
        self.tools = tools or []

    def set_system_instruction(self, text: str | None) -> None:
        self.system_instruction = text


class OpenAIBackend(_PlaceholderBackend):
    provider_name = "OpenAI"


class AnthropicBackend(_PlaceholderBackend):
    provider_name = "Anthropic"


class GeminiBackend(VLMBackend):
    """Google Gemini backend on the `google-genai` SDK.

    The new SDK is stateless per call: `client.models.generate_content` accepts
    a per-call `GenerateContentConfig` that carries the system instruction and
    the (optional) tool declarations, so we never need to rebuild a model.
    """

    def __init__(
        self,
        model_name: str,
        tools: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "Install the new SDK: `uv add google-genai` "
                "(replaces the deprecated google-generativeai)."
            ) from exc

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        self._types = types
        self.model_name = model_name
        self.system_instruction = system_instruction
        # Canonical tool specs accepted in set_tools; payload is the Gemini-shaped wrapper.
        self._tools_canonical: list[dict[str, Any]] = []
        self._tools_payload: list[dict[str, Any]] = []
        # http_options.timeout is in milliseconds in google-genai.
        timeout_ms = int(kwargs.get("timeout_ms", 300_000))
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        if tools:
            self.set_tools(tools)
        logger.info("Gemini backend ready (model=%s)", model_name)

    def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
        """Accept canonical {name, description, parameters} specs; cache the Gemini payload."""
        canonical = list(tools or [])
        if canonical == self._tools_canonical:
            return  # no-op when nothing changed
        self._tools_canonical = canonical
        self._tools_payload = (
            [{"function_declarations": canonical}] if canonical else []
        )

    def set_system_instruction(self, text: str | None) -> None:
        """Swap the system instruction in place; picked up on the next generate call."""
        self.system_instruction = text

    def _config(self) -> Any:
        # A fresh GenerateContentConfig per call — cheap; carries system + tools.
        cfg: dict[str, Any] = {}
        if self.system_instruction:
            cfg["system_instruction"] = self.system_instruction
        if self._tools_payload:
            cfg["tools"] = self._tools_payload
            cfg["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
        return self._types.GenerateContentConfig(**cfg)

    def _prepare_image(self, img: ImageInput) -> Image.Image:
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        if isinstance(img, Image.Image):
            return img
        raise ValueError(f"Unsupported image type: {type(img)}")

    def _generate(self, contents: list[Any]) -> Any:
        # Retry transient/quota errors with exponential backoff; surface other errors immediately.
        max_retries, base_delay = 5, 2.0
        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=self._config(),
                )
            except Exception as exc:
                msg = str(exc).lower()
                transient = any(
                    t in msg
                    for t in (
                        "429",
                        "quota",
                        "rate",
                        "500",
                        "502",
                        "503",
                        "504",
                        "bad gateway",
                        "deadline",
                        "unavailable",
                        "internal",
                        "timed out",
                        "timeout",
                    )
                )
                if not transient or attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "Gemini transient error (%s); retry in %.1fs", exc, delay
                )
                time.sleep(delay)
        raise RuntimeError("Gemini retry budget exhausted")  # unreachable

    def extract_usage(self, response: Any) -> dict[str, int | None] | None:
        # Normalise Gemini's usage_metadata into the backend-agnostic canonical dict.
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return None
        return {
            "prompt": getattr(um, "prompt_token_count", None),
            "output": getattr(um, "candidates_token_count", None),
            "total": getattr(um, "total_token_count", None),
            "thoughts": getattr(um, "thoughts_token_count", None),
            "cached": getattr(um, "cached_content_token_count", None),
            "tool_use": getattr(um, "tool_use_prompt_token_count", None),
        }

    @staticmethod
    def _is_safety_filtered(response: Any) -> bool:
        cands = getattr(response, "candidates", None) or []
        return bool(cands and getattr(cands[0], "finish_reason", None) == 12)

    @staticmethod
    def _extract_text(response: Any) -> str:
        # response.text is a convenience accessor; empty under tool calls,
        # so we still walk parts as a fallback.
        text = getattr(response, "text", None)
        if text:
            return str(text)
        chunks: list[str] = []
        for cand in getattr(response, "candidates", None) or []:
            for part in getattr(getattr(cand, "content", None), "parts", []) or []:
                t = getattr(part, "text", None)
                if t:
                    chunks.append(str(t))
        return " ".join(chunks)

    def get_query(
        self,
        img: ImageInput | Sequence[ImageInput],
        text: str,
        module_name: str = "Unknown",
    ) -> Any:
        imgs: list[Any] = (
            [img] if isinstance(img, (Image.Image, np.ndarray)) else list(img)
        )
        contents: list[Any] = [text, *[self._prepare_image(i) for i in imgs]]
        logger.info("[%s] Gemini query (%d parts)", module_name, len(contents))
        start = time.time()
        response = self._generate(contents)
        if self._is_safety_filtered(response):
            logger.warning(
                "[%s] safety-filtered; falling back to text-only", module_name
            )
            return self.get_text_query(text, module_name)
        logger.debug(
            "[%s] Gemini query completed in %.2fs", module_name, time.time() - start
        )
        self._log_usage(response, module_name)
        # When tools are active, callers need the structured response; otherwise return plain text.
        return response if self._tools_payload else self._extract_text(response)

    def get_text_query(self, text: str, module_name: str = "Unknown") -> Any:
        start = time.time()
        response = self._generate([text])
        if self._is_safety_filtered(response):
            return ""
        logger.debug(
            "[%s] Gemini text query completed in %.2fs",
            module_name,
            time.time() - start,
        )
        self._log_usage(response, module_name)
        return response if self._tools_payload else self._extract_text(response)

    def _log_usage(self, response: Any, module_name: str) -> None:
        # Generic wording ("LLM usage") so any future backend can reuse the line shape.
        usage = self.extract_usage(response)
        if usage is None:
            return
        logger.info(
            "[%s] LLM usage: prompt=%s output=%s total=%s (thoughts=%s, cached=%s)",
            module_name,
            usage.get("prompt"),
            usage.get("output"),
            usage.get("total"),
            usage.get("thoughts"),
            usage.get("cached"),
        )

    def _count_tokens_config(self) -> Any:
        cfg: dict[str, Any] = {}
        if self.system_instruction:
            cfg["system_instruction"] = self.system_instruction
        if self._tools_payload:
            cfg["tools"] = self._tools_payload
        if not cfg:
            return None
        return self._types.CountTokensConfig(**cfg)

    def count_input_tokens(
        self, contents: list[Any], module_name: str = "Unknown"
    ) -> int:
        try:
            response = self.client.models.count_tokens(
                model=self.model_name,
                contents=contents,
                # config=self._count_tokens_config(),
            )
            total = getattr(response, "total_tokens", None)
            if total is None:
                total = getattr(response, "totalTokens", None)
            if total is None:
                raise ValueError(f"count_tokens response missing total_tokens: {response!r}")
            return int(total)
        except Exception as exc:
            estimated = _estimate_input_tokens(contents)
            logger.warning(
                "[%s] count_tokens failed; using local estimate=%d: %s",
                module_name,
                estimated,
                exc,
            )
            return estimated

    def context_window_tokens(self) -> int | None:
        model = self.model_name.lower()
        if any(
            marker in model
            for marker in (
                "gemini-3.5",
                "gemini-3.1",
                "gemini-3",
                "gemini-2.5"
            )
        ):
            return 1_048_576
        return None

    # --- Multi-turn conversation API ------------------------------------

    def _image_part(self, img: ImageInput) -> Any:
        pil = self._prepare_image(img)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return self._types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

    def build_user_turn(self, text: str, images: Sequence[ImageInput]) -> Any:
        parts = [self._types.Part.from_text(text=text)]
        parts.extend(self._image_part(img) for img in images)
        return self._types.Content(role="user", parts=parts)

    def build_tool_results_turn(
        self, pairs: Sequence[tuple[str, dict[str, Any]]]
    ) -> Any:
        parts = [
            self._types.Part.from_function_response(name=name, response=response)
            for name, response in pairs
        ]
        return self._types.Content(role="user", parts=parts)

    def model_turn(self, response: Any) -> Any:
        cands = getattr(response, "candidates", None) or []
        content = getattr(cands[0], "content", None) if cands else None
        # Fall back to an empty model turn so the conversation stays well-formed
        # even if a response carried no candidate content.
        return content or self._types.Content(role="model", parts=[])

    @staticmethod
    def _initial_text_only_fallback(contents: list[Any]) -> str | None:
        """Return initial user-turn text when a multimodal first turn can fallback."""
        if len(contents) != 1:
            return None
        content = contents[0]
        if getattr(content, "role", None) != "user":
            return None

        texts: list[str] = []
        has_non_text_part = False
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            data = getattr(part, "data", None)
            if text is None and isinstance(data, dict):
                text = data.get("text")
            if text:
                texts.append(str(text))
            else:
                has_non_text_part = True
        if not texts or not has_non_text_part:
            return None
        return "\n".join(texts)

    def get_query_contents(
        self, contents: list[Any], module_name: str = "Unknown"
    ) -> Any:
        logger.info(
            "[%s] Gemini conversation query (%d turns)", module_name, len(contents)
        )
        start = time.time()
        response = self._generate(contents)
        if self._is_safety_filtered(response):
            fallback_text = self._initial_text_only_fallback(contents)
            if fallback_text is not None:
                logger.warning(
                    "[%s] conversation first turn safety-filtered; "
                    "falling back to text-only",
                    module_name,
                )
                response = self._generate([fallback_text])
                if self._is_safety_filtered(response):
                    logger.warning(
                        "[%s] text-only conversation fallback safety-filtered",
                        module_name,
                    )
            else:
                logger.warning("[%s] conversation turn safety-filtered", module_name)
        logger.debug(
            "[%s] Gemini conversation turn in %.2fs", module_name, time.time() - start
        )
        self._log_usage(response, module_name)
        return response


# =====================================================================
# vLLM backend — local OpenAI-compatible server for open-weight models.
#
# The continual-harness consumers (extract_function_calls, parse_action_response,
# serialize_response, the agent loop) read responses purely via getattr on a
# Gemini-shaped object. This backend therefore returns lightweight duck-typed
# objects with the SAME attribute shape so no consumer code changes are needed:
#   response.candidates[0].content.parts[i].function_call.{name,args}
#   response.candidates[0].content.parts[i].text
#   response.candidates[0].finish_reason
# Conversation turns built by build_user_turn / build_tool_results_turn /
# model_turn are the same _VLLMContent objects, converted to OpenAI chat
# messages at send time inside get_query_contents.
# =====================================================================


@dataclass(slots=True)
class _VLLMFunctionCall:
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class _VLLMPart:
    text: str | None = None
    function_call: _VLLMFunctionCall | None = None


@dataclass(slots=True)
class _VLLMContent:
    """Duck-types google-genai Content. Also carries replay state per role.

    role == "user": parts[0].text is the prompt; `images` holds PIL/ndarray.
    role == "model": `raw_message` is the OpenAI assistant message dict (with
        tool_calls + ids) to replay verbatim; parts mirror it for inspection.
    role == "tool_results": `tool_pairs` is the ordered (name, response) list.
    """

    role: str
    parts: list[_VLLMPart] = field(default_factory=list)
    images: list[ImageInput] = field(default_factory=list)
    tool_pairs: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raw_message: dict[str, Any] | None = None


@dataclass(slots=True)
class _VLLMCandidate:
    content: _VLLMContent
    finish_reason: Any = None


@dataclass(slots=True)
class _VLLMUsage:
    prompt: int | None = None
    output: int | None = None
    total: int | None = None
    thoughts: int | None = None
    cached: int | None = None
    tool_use: int | None = None


@dataclass(slots=True)
class _VLLMResponse:
    candidates: list[_VLLMCandidate]
    usage_metadata: _VLLMUsage | None = None


# --- Model registry ----------------------------------------------------------
# Single source of truth for per-model serving config. `tool_parser` and
# `reasoning_parser` are the vLLM `--tool-call-parser` / `--reasoning-parser`
# values for the model's architecture — they are a property of the MODEL, not
# the deployment, so they are NOT env-configurable; set them here. `vision`,
# `max_model_len`, and `path` are deployment knobs that may be overridden
# per-run via VLLM_SUPPORTS_VISION / VLLM_MAX_MODEL_LEN / VLLM_MODEL_PATH. Add
# an entry before serving a new model. Parser names are vLLM-version-specific —
# confirm against the installed vLLM.


@dataclass(slots=True)
class _VLLMModelSpec:
    path: str | None
    vision: bool
    tool_parser: str
    max_model_len: int
    reasoning_parser: str | None = None
    # Multi-token-prediction / speculative decoding. When `speculative_method`
    # is set, the launcher passes `--speculative-config` with a JSON
    # {"method": ..., "num_speculative_tokens": ...} (the canonical vLLM form
    # that covers both the dotted-CLI and JSON spellings). Leave as None for
    # models without an MTP head (e.g. gpt-oss).
    speculative_method: str | None = None
    num_speculative_tokens: int | None = None


_VLLM_MODEL_REGISTRY: dict[str, _VLLMModelSpec] = {
    "glm-4.7-flash": _VLLMModelSpec(
        path=None, vision=False, tool_parser="glm47",
        reasoning_parser="glm45", max_model_len=128_000,
        speculative_method="mtp", num_speculative_tokens=1,
    ),
    "qwen3.6-27b": _VLLMModelSpec(
        path=None, vision=True, tool_parser="qwen3_coder",
        reasoning_parser="qwen3", max_model_len=256_000,
        speculative_method="qwen3_next_mtp", num_speculative_tokens=2,
    ),
    # gpt-oss uses the harmony format; vLLM surfaces its tool calls via the
    # `openai` tool parser with --enable-auto-tool-choice (per the OpenAI vLLM
    # cookbook). No separate reasoning parser is needed.
    "gpt-oss-120b": _VLLMModelSpec(
        path=None, vision=False, tool_parser="openai", max_model_len=128_000,
    ),
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- Module-level vLLM server manager ----------------------------------------
# A single vLLM server is shared across ALL backend instances (orchestrator,
# subagents, evolver) and Swarm threads — never one server per game. Idempotent
# and health-check-first, guarded by a lock.

_VLLM_SERVER_LOCK = threading.Lock()
_VLLM_SERVER_PROC: subprocess.Popen[bytes] | None = None
_VLLM_SERVER_OWNED: bool = False


def _url_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            return bool(getattr(response, "status", 200) == 200)
    except (OSError, URLError):
        return False


def _server_ready(base_url: str, timeout: float = 2.0) -> bool:
    root = base_url.rstrip("/").removesuffix("/v1")
    return _url_ok(f"{root}/health", timeout=timeout) or _url_ok(
        f"{base_url.rstrip('/')}/models", timeout=timeout
    )


def _build_vllm_command(
    *,
    model_path: str,
    served_model_name: str,
    host: str,
    port: int,
    tool_parser: str,
    max_model_len: int,
    reasoning_parser: str | None,
    tensor_parallel_size: int,
    kv_cache_dtype: str,
    enforce_eager: bool,
    gpu_memory_utilization: float,
    extra_args: Sequence[str],
    speculative_method: str | None = None,
    num_speculative_tokens: int | None = None,
) -> list[str]:
    args = [
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        tool_parser,
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if reasoning_parser:
        args.extend(["--reasoning-parser", reasoning_parser])
    if speculative_method:
        # One JSON arg works for every MTP variant (GLM's `mtp`, Qwen's
        # `qwen3_next_mtp`, ...) and is equivalent to the dotted-CLI spelling.
        spec_cfg: dict[str, Any] = {"method": speculative_method}
        if num_speculative_tokens is not None:
            spec_cfg["num_speculative_tokens"] = num_speculative_tokens
        args.extend(["--speculative-config", json.dumps(spec_cfg)])
    if enforce_eager:
        args.append("--enforce-eager")
    args.extend(extra_args)

    # vLLM is installed in this same venv via the `open-vlm` extra, so the
    # `vllm` console script is normally on PATH. VLLM_SERVE_BIN is an optional
    # override (e.g. to point at a different vLLM install).
    vllm_exe = os.getenv("VLLM_SERVE_BIN") or shutil.which("vllm")
    if vllm_exe:
        return [vllm_exe, "serve", model_path, *args]
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        *args,
    ]


def start_vllm_server(
    *,
    model_path: str,
    served_model_name: str,
    base_url: str,
    host: str,
    port: int,
    tool_parser: str,
    max_model_len: int,
    reasoning_parser: str | None = None,
    tensor_parallel_size: int = 1,
    kv_cache_dtype: str = "auto",
    enforce_eager: bool = False,
    gpu_memory_utilization: float = 0.90,
    extra_args: Sequence[str] = (),
    speculative_method: str | None = None,
    num_speculative_tokens: int | None = None,
    timeout_s: int = 1800,
    poll_interval_s: float = 5.0,
) -> None:
    """Start (or reuse) a local vLLM OpenAI-compatible server; block until ready.

    Idempotent: returns immediately if a server already answers at base_url
    (whether we started it or it was launched externally). Safe to call from
    every backend instance and from multiple Swarm threads.
    """
    global _VLLM_SERVER_PROC, _VLLM_SERVER_OWNED

    with _VLLM_SERVER_LOCK:
        if _server_ready(base_url):
            return
        if _VLLM_SERVER_PROC is not None and _VLLM_SERVER_PROC.poll() is None:
            # We started something that isn't answering yet; wait on it below.
            pass
        else:
            cmd = _build_vllm_command(
                model_path=model_path,
                served_model_name=served_model_name,
                host=host,
                port=port,
                tool_parser=tool_parser,
                max_model_len=max_model_len,
                reasoning_parser=reasoning_parser,
                tensor_parallel_size=tensor_parallel_size,
                kv_cache_dtype=kv_cache_dtype,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=gpu_memory_utilization,
                extra_args=extra_args,
                speculative_method=speculative_method,
                num_speculative_tokens=num_speculative_tokens,
            )
            logger.info("Starting vLLM server: %s", " ".join(cmd))
            env = os.environ.copy()
            env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            _VLLM_SERVER_PROC = subprocess.Popen(cmd, env=env)
            _VLLM_SERVER_OWNED = True
            atexit.register(stop_vllm_server)
            logger.info("vLLM server pid=%s; waiting for ready", _VLLM_SERVER_PROC.pid)

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            proc = _VLLM_SERVER_PROC
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early with code {proc.returncode}; "
                    "check the slurm/job stderr for the startup log."
                )
            if _server_ready(base_url, timeout=2.0):
                logger.info("vLLM server ready at %s", base_url)
                return
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"Timed out after {timeout_s}s waiting for vLLM server at {base_url}."
        )


def stop_vllm_server(timeout_s: int = 30) -> None:
    """Terminate the managed vLLM subprocess if we started it."""
    global _VLLM_SERVER_PROC, _VLLM_SERVER_OWNED
    with _VLLM_SERVER_LOCK:
        proc = _VLLM_SERVER_PROC
        if not _VLLM_SERVER_OWNED or proc is None or proc.poll() is not None:
            _VLLM_SERVER_PROC = None
            _VLLM_SERVER_OWNED = False
            return
        logger.info("Stopping vLLM server pid=%s", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        _VLLM_SERVER_PROC = None
        _VLLM_SERVER_OWNED = False


class VLLMBackend(VLMBackend):
    """Backend for open-weight models served by a local vLLM OpenAI server.

    Talks to the server with the `openai` client (already a core dependency).
    USD cost is intentionally not computed for this backend — only token usage
    is recorded (see estimate_vlm_usage_cost, which returns None for non-Gemini
    models, and the agent's backend-gated pricing).
    """

    def __init__(
        self,
        model_name: str,
        tools: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The vLLM backend needs the `openai` client (a core dependency); "
                "run `uv sync`. To serve a model, install the server with "
                "`uv sync --extra open-vlm` on a GPU node."
            ) from exc

        spec = _VLLM_MODEL_REGISTRY.get(model_name.lower())
        if spec is None:
            raise ValueError(
                f"Model {model_name!r} is not in the vLLM registry. Add an entry "
                f"to _VLLM_MODEL_REGISTRY (tool_parser / reasoning_parser / vision "
                f"/ max_model_len). Known: {', '.join(sorted(_VLLM_MODEL_REGISTRY))}."
            )

        self.model_name = model_name
        self.system_instruction = system_instruction
        self._tools_canonical: list[dict[str, Any]] = []
        self._tools_payload: list[dict[str, Any]] = []

        # Model-architecture properties come from the registry (not env).
        self.tool_parser = spec.tool_parser
        self.reasoning_parser = spec.reasoning_parser
        self.speculative_method = spec.speculative_method
        self.num_speculative_tokens = spec.num_speculative_tokens
        # Deployment knobs may be overridden per-run via env.
        self.base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.host = os.getenv("VLLM_HOST", "127.0.0.1")
        self.port = _env_int("VLLM_PORT", 8000)
        self.supports_vision = _env_bool("VLLM_SUPPORTS_VISION", spec.vision)
        self.max_model_len = _env_int("VLLM_MAX_MODEL_LEN", spec.max_model_len)
        self.tool_choice = os.getenv("VLLM_TOOL_CHOICE", "auto")
        self.model_path = os.getenv("VLLM_MODEL_PATH") or spec.path or model_name
        self._timeout_s = _env_int("VLLM_STARTUP_TIMEOUT_SEC", 1800)

        if _env_bool("VLLM_AUTO_START", True):
            extra = os.getenv("VLLM_EXTRA_ARGS", "").split()
            start_vllm_server(
                model_path=self.model_path,
                served_model_name=model_name,
                base_url=self.base_url,
                host=self.host,
                port=self.port,
                tool_parser=self.tool_parser,
                max_model_len=self.max_model_len,
                reasoning_parser=self.reasoning_parser,
                tensor_parallel_size=_env_int("VLLM_TENSOR_PARALLEL_SIZE", 1),
                kv_cache_dtype=os.getenv("VLLM_KV_CACHE_DTYPE", "auto"),
                enforce_eager=_env_bool("VLLM_ENFORCE_EAGER", False),
                gpu_memory_utilization=float(
                    os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
                ),
                extra_args=extra,
                speculative_method=self.speculative_method,
                num_speculative_tokens=self.num_speculative_tokens,
                timeout_s=self._timeout_s,
            )

        timeout_ms = int(kwargs.get("timeout_ms", 300_000))
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            timeout=timeout_ms / 1000.0,
        )
        if tools:
            self.set_tools(tools)
        logger.info(
            "vLLM backend ready (model=%s, vision=%s, base_url=%s)",
            model_name,
            self.supports_vision,
            self.base_url,
        )

    # --- Config -------------------------------------------------------------

    def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
        canonical = list(tools or [])
        if canonical == self._tools_canonical:
            return
        self._tools_canonical = canonical
        self._tools_payload = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object"}),
                },
            }
            for t in canonical
        ]

    def set_system_instruction(self, text: str | None) -> None:
        self.system_instruction = text

    def context_window_tokens(self) -> int | None:
        return self.max_model_len

    # --- Image / message construction --------------------------------------

    @staticmethod
    def _to_pil(img: ImageInput) -> Image.Image:
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        if isinstance(img, Image.Image):
            return img
        raise ValueError(f"Unsupported image type: {type(img)}")

    def _data_url(self, img: ImageInput) -> str:
        pil = self._to_pil(img).convert("RGBA")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _user_message(
        self, text: str, images: Sequence[ImageInput]
    ) -> dict[str, Any]:
        # Text-only models: drop images entirely. The grid is already in the
        # prompt as hex text, so nothing structural is lost.
        if not images or not self.supports_vision:
            return {"role": "user", "content": text}
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for img in images:
            content.append(
                {"type": "image_url", "image_url": {"url": self._data_url(img)}}
            )
        return {"role": "user", "content": content}

    def _to_openai_messages(self, contents: list[Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        last_tool_calls: list[dict[str, Any]] = []
        for content in contents:
            role = getattr(content, "role", "user")
            if role == "model":
                raw = getattr(content, "raw_message", None) or {
                    "role": "assistant",
                    "content": "",
                }
                messages.append(raw)
                last_tool_calls = list(raw.get("tool_calls") or [])
            elif role == "tool_results":
                pairs = getattr(content, "tool_pairs", [])
                for i, (name, response) in enumerate(pairs):
                    tc_id = (
                        last_tool_calls[i].get("id")
                        if i < len(last_tool_calls)
                        else None
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id or f"call_{i}",
                            "name": name,
                            "content": json.dumps(response, default=str),
                        }
                    )
            else:
                parts = getattr(content, "parts", [])
                text = "".join(
                    str(getattr(p, "text", "") or "") for p in parts
                )
                images = list(getattr(content, "images", []))
                messages.append(self._user_message(text, images))
        return messages

    # --- Response wrapping --------------------------------------------------

    @staticmethod
    def _parse_args(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _wrap_response(self, completion: Any) -> _VLLMResponse:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return _VLLMResponse(
                candidates=[
                    _VLLMCandidate(content=_VLLMContent(role="model"), finish_reason=None)
                ],
                usage_metadata=self._usage_from_completion(completion),
            )
        choice = choices[0]
        msg = choice.message
        parts: list[_VLLMPart] = []
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            parts.append(_VLLMPart(text=str(reasoning)))
        if getattr(msg, "content", None):
            parts.append(_VLLMPart(text=str(msg.content)))

        tool_calls_raw: list[dict[str, Any]] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            fn = tc.function
            parts.append(
                _VLLMPart(
                    function_call=_VLLMFunctionCall(
                        name=str(fn.name), args=self._parse_args(fn.arguments)
                    )
                )
            )
            tool_calls_raw.append(
                {
                    "id": getattr(tc, "id", None) or f"call_{len(tool_calls_raw)}",
                    "type": "function",
                    "function": {
                        "name": str(fn.name),
                        "arguments": fn.arguments or "{}",
                    },
                }
            )

        raw_message: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if tool_calls_raw:
            raw_message["tool_calls"] = tool_calls_raw
        content = _VLLMContent(role="model", parts=parts, raw_message=raw_message)
        return _VLLMResponse(
            candidates=[
                _VLLMCandidate(
                    content=content,
                    finish_reason=getattr(choice, "finish_reason", None),
                )
            ],
            usage_metadata=self._usage_from_completion(completion),
        )

    @staticmethod
    def _usage_from_completion(completion: Any) -> _VLLMUsage | None:
        u = getattr(completion, "usage", None)
        if u is None:
            return None
        thoughts: int | None = None
        details = getattr(u, "completion_tokens_details", None)
        if details is not None:
            thoughts = getattr(details, "reasoning_tokens", None)
        cached: int | None = None
        pdetails = getattr(u, "prompt_tokens_details", None)
        if pdetails is not None:
            cached = getattr(pdetails, "cached_tokens", None)
        return _VLLMUsage(
            prompt=getattr(u, "prompt_tokens", None),
            output=getattr(u, "completion_tokens", None),
            total=getattr(u, "total_tokens", None),
            thoughts=thoughts,
            cached=cached,
        )

    def extract_usage(self, response: Any) -> dict[str, int | None] | None:
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return None
        return {
            "prompt": um.prompt,
            "output": um.output,
            "total": um.total,
            "thoughts": um.thoughts,
            "cached": um.cached,
            "tool_use": um.tool_use,
        }

    def _log_usage(self, response: Any, module_name: str) -> None:
        usage = self.extract_usage(response)
        if usage is None:
            return
        logger.info(
            "[%s] LLM usage: prompt=%s output=%s total=%s (thoughts=%s, cached=%s)",
            module_name,
            usage.get("prompt"),
            usage.get("output"),
            usage.get("total"),
            usage.get("thoughts"),
            usage.get("cached"),
        )

    # --- Completion call ----------------------------------------------------

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        max_retries, base_delay = 5, 2.0
        kwargs: dict[str, Any] = {"model": self.model_name, "messages": messages}
        if self._tools_payload:
            kwargs["tools"] = self._tools_payload
            kwargs["tool_choice"] = self.tool_choice
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                transient = any(
                    t in msg
                    for t in (
                        "429",
                        "rate",
                        "500",
                        "502",
                        "503",
                        "504",
                        "timeout",
                        "timed out",
                        "connection",
                        "unavailable",
                        "overloaded",
                    )
                )
                if not transient or attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning("vLLM transient error (%s); retry in %.1fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError("vLLM retry budget exhausted")  # unreachable

    def get_query(
        self,
        img: ImageInput | Sequence[ImageInput],
        text: str,
        module_name: str = "Unknown",
    ) -> Any:
        imgs: list[ImageInput] = (
            [img]
            if isinstance(img, (Image.Image, np.ndarray))
            else [i for i in (img or []) if i is not None]
        )
        messages: list[dict[str, Any]] = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        messages.append(self._user_message(text, imgs))
        completion = self._create(messages)
        response = self._wrap_response(completion)
        self._log_usage(response, module_name)
        return response if self._tools_payload else self._extract_text(response)

    def get_text_query(self, text: str, module_name: str = "Unknown") -> Any:
        messages: list[dict[str, Any]] = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        messages.append({"role": "user", "content": text})
        completion = self._create(messages)
        response = self._wrap_response(completion)
        self._log_usage(response, module_name)
        return response if self._tools_payload else self._extract_text(response)

    @staticmethod
    def _extract_text(response: _VLLMResponse) -> str:
        chunks: list[str] = []
        for cand in response.candidates:
            for part in cand.content.parts:
                if part.text:
                    chunks.append(part.text)
        return " ".join(chunks)

    # --- Multi-turn conversation API ---------------------------------------

    def build_user_turn(self, text: str, images: Sequence[ImageInput]) -> Any:
        return _VLLMContent(
            role="user", parts=[_VLLMPart(text=text)], images=list(images)
        )

    def build_tool_results_turn(
        self, pairs: Sequence[tuple[str, dict[str, Any]]]
    ) -> Any:
        return _VLLMContent(role="tool_results", tool_pairs=list(pairs))

    def model_turn(self, response: Any) -> Any:
        cands = getattr(response, "candidates", None) or []
        if cands:
            return cands[0].content
        return _VLLMContent(role="model", raw_message={"role": "assistant", "content": ""})

    def get_query_contents(
        self, contents: list[Any], module_name: str = "Unknown"
    ) -> Any:
        messages = self._to_openai_messages(contents)
        logger.info(
            "[%s] vLLM conversation query (%d turns)", module_name, len(contents)
        )
        completion = self._create(messages)
        response = self._wrap_response(completion)
        self._log_usage(response, module_name)
        return response


class VLM:
    """Provider wrapper for vision-language model backends."""

    BACKENDS: ClassVar[dict[str, type[VLMBackend]]] = {
        "gemini": GeminiBackend,
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
        "vllm": VLLMBackend,
    }

    def __init__(
        self,
        model_name: str,
        backend: str = "gemini",
        tools: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.backend_name = (
            self._auto_detect_backend(model_name) if backend == "auto" else backend
        ).lower()
        if self.backend_name not in self.BACKENDS:
            raise ValueError(
                f"Unsupported VLM backend '{backend}'. "
                f"Available: {', '.join(sorted(self.BACKENDS))}"
            )

        backend_cls: Any = self.BACKENDS[self.backend_name]
        self.backend = cast(
            VLMBackend,
            backend_cls(
                model_name,
                tools=tools,
                system_instruction=system_instruction,
                **kwargs,
            ),
        )

    @staticmethod
    def _auto_detect_backend(model_name: str) -> str:
        """Guess the provider from the model name; used when backend='auto'."""
        lowered = model_name.lower()
        if "gemini" in lowered or "palm" in lowered:
            return "gemini"
        if "claude" in lowered:
            return "anthropic"
        if (
            lowered.startswith("gpt")
            or lowered.startswith("o3")
            or lowered.startswith("o4")
            or lowered.startswith("o5")
            or "openai" in lowered
        ):
            return "openai"
        raise ValueError(f"Could not auto-detect VLM backend for model '{model_name}'")

    def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
        self.backend.set_tools(tools)

    def set_system_instruction(self, text: str | None) -> None:
        self.backend.set_system_instruction(text)

    def extract_usage(self, response: Any) -> dict[str, int | None] | None:
        return self.backend.extract_usage(response)

    def get_query(
        self,
        img: ImageInput | Sequence[ImageInput],
        text: str,
        module_name: str = "Unknown",
    ) -> Any:
        return self.backend.get_query(img, text, module_name)

    def get_text_query(self, text: str, module_name: str = "Unknown") -> Any:
        return self.backend.get_text_query(text, module_name)

    # --- Multi-turn conversation API (forwarded to the backend) ---------

    def build_user_turn(self, text: str, images: Sequence[ImageInput]) -> Any:
        return self.backend.build_user_turn(text, images)

    def build_tool_results_turn(
        self, pairs: Sequence[tuple[str, dict[str, Any]]]
    ) -> Any:
        return self.backend.build_tool_results_turn(pairs)

    def model_turn(self, response: Any) -> Any:
        return self.backend.model_turn(response)

    def get_query_contents(
        self, contents: list[Any], module_name: str = "Unknown"
    ) -> Any:
        return self.backend.get_query_contents(contents, module_name)

    def count_input_tokens(
        self, contents: list[Any], module_name: str = "Unknown"
    ) -> int:
        return self.backend.count_input_tokens(contents, module_name)

    def context_window_tokens(self) -> int | None:
        return self.backend.context_window_tokens()
