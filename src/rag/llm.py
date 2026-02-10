"""Mistral LLM wrapper with retry/backoff and test doubles."""

from __future__ import annotations

import inspect
import logging
import os
import time
from typing import Any, Protocol


LOGGER = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        timeout_seconds: int,
    ) -> str:
        ...


def _extract_text_from_response(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content
        content = response.get("content", "")
        if isinstance(content, str):
            return content
    if hasattr(response, "choices"):
        choices = getattr(response, "choices") or []
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message is not None:
                content = getattr(message, "content", "")
                if isinstance(content, list):
                    # Newer SDK can return a structured list
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if text:
                                text_parts.append(str(text))
                        else:
                            text = getattr(item, "text", None)
                            if text:
                                text_parts.append(str(text))
                    if text_parts:
                        return "\n".join(text_parts)
                if isinstance(content, str):
                    return content
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    return str(response)


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(token in message for token in ("429", "rate limit", "timeout", "temporarily"))


class MistralLLMClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "mistral-small-latest",
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
        logger: logging.Logger | None = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("MISTRAL_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is missing.")
        self.model = model
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.logger = logger or LOGGER
        self.client = self._init_client()

    def _init_client(self) -> Any:
        # New SDK
        try:
            from mistralai import Mistral  # type: ignore

            return Mistral(api_key=self.api_key)
        except Exception:
            pass

        # Old SDK
        from mistralai.client import MistralClient  # type: ignore

        return MistralClient(api_key=self.api_key)

    def _call_chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        timeout_seconds: int,
    ) -> Any:
        # New SDK style: client.chat.complete(...)
        if hasattr(self.client, "chat") and hasattr(self.client.chat, "complete"):
            method = self.client.chat.complete
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            signature = inspect.signature(method)
            if "timeout_ms" in signature.parameters:
                kwargs["timeout_ms"] = int(timeout_seconds * 1000)
            elif "timeout" in signature.parameters:
                kwargs["timeout"] = timeout_seconds
            return method(**kwargs)

        # Old SDK style: client.chat(...)
        method = getattr(self.client, "chat")
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        signature = inspect.signature(method)
        if "timeout" in signature.parameters:
            kwargs["timeout"] = timeout_seconds
        return method(**kwargs)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        timeout_seconds: int,
    ) -> str:
        prompt_chars = len(system_prompt) + len(user_prompt)
        self.logger.info(
            "llm.generate model=%s prompt_chars=%s",
            self.model,
            prompt_chars,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._call_chat(
                    messages=messages,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                )
                text = _extract_text_from_response(response).strip()
                if not text:
                    raise RuntimeError("Empty response from Mistral.")
                return text
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                retryable = _is_retryable_error(exc)
                self.logger.warning(
                    "llm.generate failed attempt=%s/%s retryable=%s error=%s",
                    attempt,
                    self.max_retries,
                    retryable,
                    exc,
                )
                if attempt >= self.max_retries or not retryable:
                    break
                time.sleep(self.backoff_seconds * attempt)
        raise RuntimeError(f"Mistral generation failed: {last_error}") from last_error


class FakeLLMClient:
    """Deterministic LLM stub used in tests/offline smoke evaluation."""

    def __init__(self, fixed_answer: str | None = None) -> None:
        self.fixed_answer = fixed_answer

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        timeout_seconds: int,
    ) -> str:
        del system_prompt, temperature, timeout_seconds
        if self.fixed_answer:
            return self.fixed_answer
        if "Je ne peux pas répondre avec certitude à partir des données disponibles." in user_prompt:
            return "Je ne peux pas répondre avec certitude à partir des données disponibles."
        return (
            "Synthese: recommandations basees sur les evenements recuperes.\n"
            "- Voir les sources pour titres, dates, lieux et URL.\n"
            "Pourquoi ces choix ? Pertinence semantique avec la question."
        )
