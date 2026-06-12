from __future__ import annotations

import io
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Sequence, TypeAlias, cast

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ImageInput: TypeAlias = Image.Image | np.ndarray


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
                config=self._count_tokens_config(),
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


class VLM:
    """Provider wrapper for vision-language model backends."""

    BACKENDS: ClassVar[dict[str, type[VLMBackend]]] = {
        "gemini": GeminiBackend,
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
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
