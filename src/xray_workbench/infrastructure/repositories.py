from datetime import datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import JSON, DateTime, String, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from xray_workbench.domain.detection import BoundingBox, Detection
from xray_workbench.domain.inspection import Inspection, InspectionStatus
from xray_workbench.domain.review import OperatorReview, ReviewDecision
from xray_workbench.domain.risk import RiskLevel
from xray_workbench.infrastructure.database import Base


class InspectionRepository(Protocol):
    def save(self, inspection: Inspection) -> None: ...

    def get(self, inspection_id: UUID) -> Inspection | None: ...

    def list_recent(self, limit: int) -> list[Inspection]: ...


class InspectionRecord(Base):
    __tablename__ = "inspections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    image_id: Mapped[str] = mapped_column(String(255), index=True)
    status: Mapped[str] = mapped_column(String(64), index=True)
    detections: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    review: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)


class SqlAlchemyInspectionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, inspection: Inspection) -> None:
        record = InspectionRecord(
            id=str(inspection.id),
            image_id=inspection.image_id,
            status=inspection.status.value,
            detections=[_detection_to_dict(item) for item in inspection.detections],
            model_version=inspection.model_version,
            created_at=inspection.created_at,
            reviewed_at=inspection.reviewed_at,
            review=_review_to_dict(inspection.review),
        )
        self._session.merge(record)
        self._session.commit()

    def get(self, inspection_id: UUID) -> Inspection | None:
        record = self._session.scalar(
            select(InspectionRecord).where(InspectionRecord.id == str(inspection_id))
        )
        if record is None:
            return None
        return _record_to_domain(record)

    def list_recent(self, limit: int) -> list[Inspection]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        records = self._session.scalars(
            select(InspectionRecord)
            .order_by(InspectionRecord.created_at.desc())
            .limit(limit)
        ).all()
        return [_record_to_domain(record) for record in records]


def _record_to_domain(record: InspectionRecord) -> Inspection:
    return Inspection(
        id=UUID(record.id),
        image_id=record.image_id,
        status=InspectionStatus(record.status),
        detections=[_detection_from_dict(item) for item in record.detections],
        model_version=record.model_version,
        created_at=record.created_at,
        reviewed_at=record.reviewed_at,
        review=_review_from_dict(record.review),
    )


def _detection_to_dict(detection: Detection) -> dict[str, object]:
    box = detection.bounding_box
    return {
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "risk_level": detection.risk_level.value,
        "bounding_box": {"x": box.x, "y": box.y, "width": box.width, "height": box.height},
    }


def _detection_from_dict(value: dict[str, object]) -> Detection:
    box = value["bounding_box"]
    if not isinstance(box, dict):
        raise ValueError("Stored bounding box is invalid")
    return Detection(
        class_name=str(value["class_name"]),
        confidence=_as_float(value["confidence"]),
        risk_level=RiskLevel(str(value["risk_level"])),
        bounding_box=BoundingBox(
            x=_as_float(box["x"]),
            y=_as_float(box["y"]),
            width=_as_float(box["width"]),
            height=_as_float(box["height"]),
        ),
    )


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError("Stored numeric value is invalid")


def _review_to_dict(review: OperatorReview | None) -> dict[str, object] | None:
    if review is None:
        return None
    return {
        "decision": review.decision.value,
        "notes": review.notes,
        "reviewed_at": review.reviewed_at.isoformat(),
    }


def _review_from_dict(value: dict[str, object] | None) -> OperatorReview | None:
    if value is None:
        return None
    notes = value.get("notes")
    return OperatorReview(
        decision=ReviewDecision(str(value["decision"])),
        notes=str(notes) if notes is not None else None,
        reviewed_at=datetime.fromisoformat(str(value["reviewed_at"])),
    )
