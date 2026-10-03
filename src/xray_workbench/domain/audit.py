from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4


class AuditEventType(StrEnum):
    IMAGE_RECEIVED = "image_received"
    PROCESSING_STARTED = "processing_started"
    DETECTION_COMPLETED = "detection_completed"
    MODEL_UNAVAILABLE = "model_unavailable"
    REVIEW_RECORDED = "review_recorded"


@dataclass(frozen=True, slots=True)
class AuditEvent:
    inspection_id: UUID
    event_type: AuditEventType
    message: str
    details: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
