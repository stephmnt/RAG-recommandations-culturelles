"""Lightweight API exceptions shared across modules."""

from __future__ import annotations


class RebuildInProgressError(RuntimeError):
    """Raised when a rebuild request is received while another is running."""


class IndexUnavailableError(RuntimeError):
    """Raised when index artifacts are unavailable or not loaded."""


__all__ = ["RebuildInProgressError", "IndexUnavailableError"]
