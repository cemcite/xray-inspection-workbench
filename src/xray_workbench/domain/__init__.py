"""Framework-independent domain model."""

from xray_workbench.domain.audit import AuditEvent, AuditEventType
from xray_workbench.domain.detection import BoundingBox, Detection
from xray_workbench.domain.inspection import Inspection, InspectionStatus
from xray_workbench.domain.review import OperatorReview, ReviewDecision
from xray_workbench.domain.risk import RiskLevel

__all__ = [
    "AuditEvent",
    "AuditEventType",
    "BoundingBox",
    "Detection",
    "Inspection",
    "InspectionStatus",
    "OperatorReview",
    "ReviewDecision",
    "RiskLevel",
]
