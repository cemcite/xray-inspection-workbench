from uuid import UUID

from xray_workbench.application.risk_service import RiskClassifier
from xray_workbench.domain.audit import AuditEvent, AuditEventType
from xray_workbench.domain.detection import Detection
from xray_workbench.domain.inspection import Inspection, InspectionStatus
from xray_workbench.domain.review import OperatorReview
from xray_workbench.infrastructure.audit_repository import AuditRepository
from xray_workbench.infrastructure.image_storage import (
    ImageReferenceRepository,
    ImageStore,
    StoredImage,
)
from xray_workbench.infrastructure.repositories import InspectionRepository
from xray_workbench.vision.detector import Detector, ImageArray


class InspectionNotFoundError(LookupError):
    pass


class InspectionImageNotFoundError(LookupError):
    pass


class InspectionService:
    def __init__(
        self,
        detector: Detector,
        classifier: RiskClassifier,
        repository: InspectionRepository,
        audit_repository: AuditRepository | None = None,
        image_store: ImageStore | None = None,
        image_repository: ImageReferenceRepository | None = None,
    ) -> None:
        self._detector = detector
        self._classifier = classifier
        self._repository = repository
        self._audit_repository = audit_repository
        self._image_store = image_store
        self._image_repository = image_repository

    def inspect(
        self,
        image_id: str,
        image: ImageArray,
        *,
        image_payload: bytes | None = None,
        content_type: str = "image/png",
    ) -> Inspection:
        inspection = Inspection(image_id=image_id)
        stored_image: StoredImage | None = None
        if image_payload is not None and self._image_store is not None:
            storage_key = self._image_store.save(inspection.id, image_payload, content_type)
            stored_image = StoredImage(
                inspection_id=inspection.id,
                storage_key=storage_key,
                original_filename=image_id,
                content_type=content_type,
            )
        inspection.mark_processing()
        try:
            result = self._detector.detect(image)
        except RuntimeError:
            inspection.mark_manual_review_required()
        else:
            detections = [
                Detection(
                    class_name=candidate.class_name,
                    confidence=candidate.confidence,
                    bounding_box=candidate.bounding_box,
                    risk_level=self._classifier.classify(
                        candidate.class_name, candidate.confidence
                    ),
                )
                for candidate in result.detections
            ]
            inspection.complete(detections, result.model_version)
        self._repository.save(inspection)
        if stored_image is not None and self._image_repository is not None:
            self._image_repository.save(stored_image)
        self._record_inspection_events(inspection, stored_image is not None)
        return inspection

    def get(self, inspection_id: UUID) -> Inspection:
        inspection = self._repository.get(inspection_id)
        if inspection is None:
            raise InspectionNotFoundError(str(inspection_id))
        return inspection

    def list_recent(self, limit: int = 50) -> list[Inspection]:
        return self._repository.list_recent(limit)

    def get_image(self, inspection_id: UUID) -> tuple[StoredImage, bytes]:
        self.get(inspection_id)
        if self._image_repository is None or self._image_store is None:
            raise InspectionImageNotFoundError(str(inspection_id))
        stored = self._image_repository.get(inspection_id)
        if stored is None:
            raise InspectionImageNotFoundError(str(inspection_id))
        try:
            payload = self._image_store.read(stored.storage_key)
        except FileNotFoundError as error:
            raise InspectionImageNotFoundError(str(inspection_id)) from error
        return stored, payload

    def list_events(self, inspection_id: UUID) -> list[AuditEvent]:
        self.get(inspection_id)
        if self._audit_repository is None:
            return []
        return self._audit_repository.list_for_inspection(inspection_id)

    def review(self, inspection_id: UUID, review: OperatorReview) -> Inspection:
        inspection = self.get(inspection_id)
        inspection.apply_review(review)
        self._repository.save(inspection)
        self._append_event(
            AuditEvent(
                inspection_id=inspection.id,
                event_type=AuditEventType.REVIEW_RECORDED,
                message=f"Operator review recorded: {review.decision.value}",
                details={"decision": review.decision.value, "notes": review.notes},
            )
        )
        return inspection

    def _record_inspection_events(self, inspection: Inspection, image_stored: bool) -> None:
        self._append_event(
            AuditEvent(
                inspection_id=inspection.id,
                event_type=AuditEventType.IMAGE_RECEIVED,
                message="Image received",
                details={"image_id": inspection.image_id, "stored": image_stored},
            )
        )
        self._append_event(
            AuditEvent(
                inspection_id=inspection.id,
                event_type=AuditEventType.PROCESSING_STARTED,
                message="Inspection processing started",
            )
        )
        if inspection.status is InspectionStatus.MANUAL_REVIEW_REQUIRED:
            self._append_event(
                AuditEvent(
                    inspection_id=inspection.id,
                    event_type=AuditEventType.MODEL_UNAVAILABLE,
                    message="AI model unavailable; manual review required",
                )
            )
        else:
            self._append_event(
                AuditEvent(
                    inspection_id=inspection.id,
                    event_type=AuditEventType.DETECTION_COMPLETED,
                    message="Detection completed",
                    details={
                        "detection_count": len(inspection.detections),
                        "model_version": inspection.model_version,
                    },
                )
            )

    def _append_event(self, event: AuditEvent) -> None:
        if self._audit_repository is not None:
            self._audit_repository.append(event)
