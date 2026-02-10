"""Compatibility shim for `from mistral import MistralClient`."""

from __future__ import annotations

try:
    from mistralai.client import MistralClient  # type: ignore
except Exception:
    try:
        from mistralai import Mistral as MistralClient  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Mistral SDK not available. Install `mistralai` to use MistralClient."
        ) from exc

__all__ = ["MistralClient"]
