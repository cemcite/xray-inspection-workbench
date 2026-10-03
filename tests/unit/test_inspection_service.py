import numpy as np

from xray_workbench.application.inspection_service import InspectionService
from xray_workbench.application.risk_service import RiskClassifier, RiskConfig, RiskThresholds
from xray_workbench.domain.detection import BoundingBox
from xray_workbench.domain.inspection import Inspection, InspectionStatus
from xray_workbench.domain.risk import RiskLevel
from xray_workbench.vision.detector import DetectionCandidate
from xray_workbench.vision.fake_detector import FakeDetector


class InMemoryRepository:
    def __init__(self) -> None:
        self.items: dict[object, Inspection] = {}

    def save(self, inspection: Inspection) -> None:
        self.items[inspection.id] = inspection

    def get(self, inspection_id: object) -> Inspection | None:
        return self.items.get(inspection_id)


def test_service_maps_detector_candidate_to_domain_risk() -> None:
    detector = FakeDetector(
        candidates=(
            DetectionCandidate("knife", 0.9, BoundingBox(0.1, 0.1, 0.2, 0.3)),
        )
    )
    classifier = RiskClassifier(
        RiskConfig(
            default=RiskThresholds(0.5, 0.8),
            classes={},
        )
    )
    repository = InMemoryRepository()
    service = InspectionService(detector, classifier, repository)

    inspection = service.inspect("sample.png", np.zeros((8, 8, 3), dtype=np.uint8))

    assert inspection.status is InspectionStatus.REQUIRES_REVIEW
    assert inspection.detections[0].risk_level is RiskLevel.HIGH


def test_unavailable_detector_fails_safe_to_manual_review() -> None:
    detector = FakeDetector(ready=False)
    classifier = RiskClassifier(RiskConfig(RiskThresholds(0.5, 0.8), {}))
    service = InspectionService(detector, classifier, InMemoryRepository())

    inspection = service.inspect("sample.png", np.zeros((8, 8, 3), dtype=np.uint8))

    assert inspection.status is InspectionStatus.MANUAL_REVIEW_REQUIRED
