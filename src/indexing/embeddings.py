"""Embeddings factory with provider/fallback strategy."""

from __future__ import annotations

import logging
import os
from typing import Any


LOGGER = logging.getLogger(__name__)


DEFAULT_HF_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_MISTRAL_MODEL = "mistral-embed"


def _import_hf_embeddings_class():
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings
    except Exception:
        from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore

        return HuggingFaceEmbeddings


def _build_hf_embeddings(model_name: str, logger: logging.Logger) -> Any:
    embedding_cls = _import_hf_embeddings_class()
    logger.info("Using HuggingFace embeddings model=%s", model_name)
    return embedding_cls(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _try_build_mistral_embeddings(
    *,
    model_name: str,
    api_key: str,
    logger: logging.Logger,
) -> Any | None:
    import_errors: list[str] = []
    candidate_classes: list[type] = []

    for import_stmt, symbol in (
        ("langchain_mistralai", "MistralAIEmbeddings"),
        ("langchain_community.embeddings", "MistralAIEmbeddings"),
        ("langchain.embeddings", "MistralAIEmbeddings"),
    ):
        try:
            module = __import__(import_stmt, fromlist=[symbol])
            candidate = getattr(module, symbol)
            candidate_classes.append(candidate)
        except Exception as exc:
            import_errors.append(f"{import_stmt}.{symbol}: {exc}")

    if not candidate_classes:
        logger.warning(
            "Mistral embeddings class unavailable (%s). Falling back to HuggingFace.",
            "; ".join(import_errors),
        )
        return None

    for candidate in candidate_classes:
        try:
            logger.info("Using Mistral embeddings model=%s class=%s", model_name, candidate)
            return candidate(model=model_name, api_key=api_key)
        except Exception as exc:
            logger.warning(
                "Failed to initialize mistral embeddings with %s: %s",
                candidate,
                exc,
            )

    logger.warning("No usable Mistral embeddings implementation found. Falling back to HuggingFace.")
    return None


def get_embedding_model(
    *,
    config: dict[str, Any] | None = None,
    provider: str | None = None,
    huggingface_model: str | None = None,
    mistral_model: str | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """
    Build embedding model with a stable provider strategy.

    Priority:
      1) Explicit function args
      2) Environment variables
      3) YAML config
      4) Defaults
    """

    logger = logger or LOGGER
    config = config or {}
    embeddings_cfg = config.get("embeddings", {}) if isinstance(config, dict) else {}

    resolved_provider = (
        provider
        or os.getenv("EMBEDDING_PROVIDER")
        or embeddings_cfg.get("provider")
        or "huggingface"
    ).strip().lower()
    resolved_hf_model = (
        huggingface_model
        or os.getenv("EMBEDDING_MODEL")
        or embeddings_cfg.get("huggingface_model")
        or DEFAULT_HF_MODEL
    ).strip()
    resolved_mistral_model = (
        mistral_model
        or os.getenv("MISTRAL_EMBEDDING_MODEL")
        or embeddings_cfg.get("mistral_model")
        or DEFAULT_MISTRAL_MODEL
    ).strip()

    if resolved_provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "EMBEDDING_PROVIDER=mistral but MISTRAL_API_KEY is missing. "
                "Falling back to HuggingFace embeddings."
            )
            return _build_hf_embeddings(resolved_hf_model, logger=logger)

        mistral_embeddings = _try_build_mistral_embeddings(
            model_name=resolved_mistral_model,
            api_key=api_key,
            logger=logger,
        )
        if mistral_embeddings is not None:
            return mistral_embeddings
        return _build_hf_embeddings(resolved_hf_model, logger=logger)

    return _build_hf_embeddings(resolved_hf_model, logger=logger)
