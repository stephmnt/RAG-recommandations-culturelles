"""Schema definition for cleaned OpenAgenda events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

EVENT_RECORD_FIELDS = [
    "event_id",
    "title",
    "description",
    "start_datetime",
    "end_datetime",
    "city",
    "region",
    "department",
    "location_name",
    "address",
    "latitude",
    "longitude",
    "url",
    "tags",
    "source",
    "document_text",
    "retrieval_metadata",
]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_iso_datetime(value: str) -> bool:
    if not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _build_document_text(
    *,
    title: str,
    description: str,
    location_name: str,
    city: str,
    start_datetime: str,
) -> str:
    parts = [
        f"Titre: {title}" if title else "",
        f"Description: {description}" if description else "",
        f"Lieu: {location_name}" if location_name else "",
        f"Ville: {city}" if city else "",
        f"Date debut: {start_datetime}" if start_datetime else "",
    ]
    return "\n".join(part for part in parts if part).strip()


@dataclass
class EventRecord:
    event_id: str
    title: str
    description: str = ""
    start_datetime: str = ""
    end_datetime: str = ""
    city: str = ""
    region: str = ""
    department: str = ""
    location_name: str = ""
    address: str = ""
    latitude: float | None = None
    longitude: float | None = None
    url: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = "openagenda"
    document_text: str = ""
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_id = _clean_text(self.event_id)
        self.title = _clean_text(self.title)
        self.description = _clean_text(self.description)
        self.start_datetime = _clean_text(self.start_datetime)
        self.end_datetime = _clean_text(self.end_datetime)
        self.city = _clean_text(self.city)
        self.region = _clean_text(self.region)
        self.department = _clean_text(self.department)
        self.location_name = _clean_text(self.location_name)
        self.address = _clean_text(self.address)
        self.url = _clean_text(self.url)
        self.source = _clean_text(self.source) or "openagenda"
        self.latitude = _to_optional_float(self.latitude)
        self.longitude = _to_optional_float(self.longitude)

        if isinstance(self.tags, str):
            self.tags = [token.strip() for token in self.tags.split(",") if token.strip()]
        elif self.tags is None:
            self.tags = []
        else:
            self.tags = [_clean_text(token) for token in self.tags if _clean_text(token)]

        if not self.document_text:
            self.document_text = _build_document_text(
                title=self.title,
                description=self.description,
                location_name=self.location_name,
                city=self.city,
                start_datetime=self.start_datetime,
            )
        self.document_text = _clean_text(self.document_text)

        if not self.event_id:
            raise ValueError("event_id is required.")
        if not self.title:
            raise ValueError("title is required.")
        if not _is_iso_datetime(self.start_datetime):
            raise ValueError("start_datetime must be a valid ISO 8601 datetime.")
        if self.end_datetime and not _is_iso_datetime(self.end_datetime):
            raise ValueError("end_datetime must be a valid ISO 8601 datetime when provided.")
        if not self.document_text:
            raise ValueError("document_text cannot be empty.")

        if not isinstance(self.retrieval_metadata, dict):
            raise ValueError("retrieval_metadata must be a dict.")
        metadata = dict(self.retrieval_metadata)
        defaults = {
            "event_id": self.event_id,
            "city": self.city,
            "region": self.region,
            "department": self.department,
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
            "source": self.source,
            "url": self.url,
            "tags": self.tags,
        }
        for key, value in defaults.items():
            if key not in metadata and value not in (None, "", []):
                metadata[key] = value
        self.retrieval_metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {field_name: payload[field_name] for field_name in EVENT_RECORD_FIELDS}


def validate_record(payload: dict[str, Any]) -> EventRecord:
    """Validate and normalize a raw cleaned record."""

    return EventRecord(**payload)
