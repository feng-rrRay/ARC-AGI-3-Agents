from __future__ import annotations

import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Sequence, TypeAlias, cast

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ImageInput: TypeAlias = Image.Image | np.ndarray


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
        # Upscale 4x with NEAREST to preserve crisp pixel boundaries — each cell
        # becomes a 4x4 block of identical colour rather than getting smoothed.
        if isinstance(img, np.ndarray):
            image = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            image = img
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        w, h = image.size
        return image.resize((w * 4, h * 4), Image.Resampling.NEAREST)

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
