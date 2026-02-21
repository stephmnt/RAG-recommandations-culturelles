"""Custom exceptions for API error handling."""

from __future__ import annotations

from typing import Any


class APIError(Exception):
    """Operational API error converted to JSON response."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class IndexUnavailableError(APIError):
    def __init__(self, message: str) -> None:
        super().__init__(
            code="INDEX_UNAVAILABLE",
            message=message,
            status_code=503,
        )


class RebuildBusyError(APIError):
    def __init__(self) -> None:
        super().__init__(
            code="REBUILD_IN_PROGRESS",
            message="An index rebuild is already running.",
            status_code=409,
        )
