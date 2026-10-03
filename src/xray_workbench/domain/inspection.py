from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from xray_workbench.domain.detection import Detection
from xray_workbench.domain.review import OperatorReview, ReviewDecision


class InspectionStatus(StrEnum):
    CREATED = "created"
    PROCESSING = "processing"
    REQUIRES_REVIEW = "requires_review"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    CLEARED = "cleared"
    THREAT_CONFIRMED = "threat_confirmed"
    FAILED = "failed"


_REVIEWABLE_STATUSES = {
    InspectionStatus.REQUIRES_REVIEW,
    InspectionStatus.MANUAL_REVIEW_REQUIRED,
}


@dataclass(slots=True)
class Inspection:
    image_id: str
    id: UUID = field(default_factory=uuid4)
    status: InspectionStatus = InspectionStatus.CREATED
    detections: list[Detection] = field(default_factory=list)
    model_version: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reviewed_at: datetime | None = None
    review: OperatorReview | None = None

    def mark_processing(self) -> None:
        if self.status is not InspectionStatus.CREATED:
            raise ValueError("Only a created inspection can begin processing")
        self.status = InspectionStatus.PROCESSING

    def complete(self, detections: list[Detection], model_version: str) -> None:
        if self.status is not InspectionStatus.PROCESSING:
            raise ValueError("Only a processing inspection can be completed")
        self.detections = list(detections)
        self.model_version = model_version
        self.status = (
            InspectionStatus.REQUIRES_REVIEW if detections else InspectionStatus.CLEARED
        )

    def mark_manual_review_required(self) -> None:
        if self.status not in {InspectionStatus.CREATED, InspectionStatus.PROCESSING}:
            raise ValueError("Manual review can only replace a pending model decision")
        self.status = InspectionStatus.MANUAL_REVIEW_REQUIRED

    def apply_review(self, review: OperatorReview) -> None:
        if self.status not in _REVIEWABLE_STATUSES:
            raise ValueError(f"Inspection in '{self.status}' state cannot be reviewed")
        self.review = review
        self.reviewed_at = review.reviewed_at
        if review.decision is ReviewDecision.CONFIRMED_THREAT:
            self.status = InspectionStatus.THREAT_CONFIRMED
        elif review.decision is ReviewDecision.FALSE_POSITIVE:
            self.status = InspectionStatus.CLEARED
        else:
            self.status = InspectionStatus.REQUIRES_REVIEW
