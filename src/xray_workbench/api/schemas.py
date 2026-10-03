from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from xray_workbench.domain.audit import AuditEvent, AuditEventType
from xray_workbench.domain.detection import BoundingBox, Detection
from xray_workbench.domain.inspection import Inspection, InspectionStatus
from xray_workbench.domain.review import OperatorReview, ReviewDecision
from xray_workbench.domain.risk import RiskLevel


class ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class BoundingBoxResponse(ApiModel):
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_domain(cls, box: BoundingBox) -> "BoundingBoxResponse":
        return cls(x=box.x, y=box.y, width=box.width, height=box.height)


class DetectionResponse(ApiModel):
    class_name: str = Field(alias="className")
    confidence: float
    risk_level: RiskLevel = Field(alias="riskLevel")
    bounding_box: BoundingBoxResponse = Field(alias="boundingBox")

    @classmethod
    def from_domain(cls, detection: Detection) -> "DetectionResponse":
        return cls(
            class_name=detection.class_name,
            confidence=detection.confidence,
            risk_level=detection.risk_level,
            bounding_box=BoundingBoxResponse.from_domain(detection.bounding_box),
        )


class ReviewResponse(ApiModel):
    decision: ReviewDecision
    notes: str | None
    reviewed_at: datetime = Field(alias="reviewedAt")

    @classmethod
    def from_domain(cls, review: OperatorReview) -> "ReviewResponse":
        return cls(
            decision=review.decision,
            notes=review.notes,
            reviewed_at=review.reviewed_at,
        )


class InspectionResponse(ApiModel):
    inspection_id: UUID = Field(alias="inspectionId")
    image_id: str = Field(alias="imageId")
    status: InspectionStatus
    model_version: str | None = Field(alias="modelVersion")
    detections: list[DetectionResponse]
    created_at: datetime = Field(alias="createdAt")
    reviewed_at: datetime | None = Field(alias="reviewedAt")
    review: ReviewResponse | None

    @classmethod
    def from_domain(cls, inspection: Inspection) -> "InspectionResponse":
        return cls(
            inspection_id=inspection.id,
            image_id=inspection.image_id,
            status=inspection.status,
            model_version=inspection.model_version,
            detections=[DetectionResponse.from_domain(item) for item in inspection.detections],
            created_at=inspection.created_at,
            reviewed_at=inspection.reviewed_at,
            review=ReviewResponse.from_domain(inspection.review) if inspection.review else None,
        )


class ReviewRequest(ApiModel):
    decision: ReviewDecision
    notes: str | None = Field(default=None, max_length=2000)


class AuditEventResponse(ApiModel):
    event_id: UUID = Field(alias="eventId")
    event_type: AuditEventType = Field(alias="eventType")
    message: str
    details: dict[str, str | int | float | bool | None]
    occurred_at: datetime = Field(alias="occurredAt")

    @classmethod
    def from_domain(cls, event: AuditEvent) -> "AuditEventResponse":
        return cls(
            event_id=event.id,
            event_type=event.event_type,
            message=event.message,
            details=event.details,
            occurred_at=event.occurred_at,
        )


class ModelHealth(ApiModel):
    status: str
    name: str
    version: str


class HealthResponse(ApiModel):
    status: str
    model: ModelHealth
    database: str
