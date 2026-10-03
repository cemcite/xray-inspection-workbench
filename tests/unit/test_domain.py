from uuid import uuid4

import pytest

from xray_workbench.domain.detection import BoundingBox, Detection
from xray_workbench.domain.inspection import Inspection, InspectionStatus
from xray_workbench.domain.review import OperatorReview, ReviewDecision
from xray_workbench.domain.risk import RiskLevel


def test_bounding_box_must_be_normalized_and_inside_image() -> None:
    with pytest.raises(ValueError, match="image bounds"):
        BoundingBox(x=0.9, y=0.1, width=0.2, height=0.2)


def test_detection_confidence_is_bounded() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Detection(
            class_name="knife",
            confidence=1.1,
            bounding_box=BoundingBox(0.1, 0.1, 0.2, 0.2),
            risk_level=RiskLevel.HIGH,
        )


def test_completed_detection_requires_operator_review() -> None:
    inspection = Inspection(id=uuid4(), image_id="sample.png")
    inspection.mark_processing()
    inspection.complete(
        [
            Detection(
                class_name="knife",
                confidence=0.92,
                bounding_box=BoundingBox(0.1, 0.2, 0.3, 0.4),
                risk_level=RiskLevel.HIGH,
            )
        ],
        model_version="0.1.0",
    )

    assert inspection.status is InspectionStatus.REQUIRES_REVIEW


def test_false_positive_review_clears_inspection() -> None:
    inspection = Inspection(image_id="sample.png")
    inspection.mark_processing()
    inspection.complete(
        [
            Detection(
                class_name="knife",
                confidence=0.75,
                bounding_box=BoundingBox(0.1, 0.1, 0.2, 0.2),
                risk_level=RiskLevel.HIGH,
            )
        ],
        model_version="0.1.0",
    )

    inspection.apply_review(OperatorReview(decision=ReviewDecision.FALSE_POSITIVE))

    assert inspection.status is InspectionStatus.CLEARED
    assert inspection.reviewed_at is not None


def test_terminal_inspection_cannot_be_reviewed_again() -> None:
    inspection = Inspection(image_id="sample.png", status=InspectionStatus.CLEARED)

    with pytest.raises(ValueError, match="cannot be reviewed"):
        inspection.apply_review(OperatorReview(decision=ReviewDecision.CONFIRMED_THREAT))
