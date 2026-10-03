from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from xray_workbench.application.inspection_service import InspectionService
from xray_workbench.application.risk_service import RiskClassifier, load_risk_config
from xray_workbench.infrastructure.audit_repository import SqlAlchemyAuditRepository
from xray_workbench.infrastructure.database import get_session
from xray_workbench.infrastructure.image_storage import (
    LocalImageStore,
    SqlAlchemyImageReferenceRepository,
)
from xray_workbench.infrastructure.repositories import SqlAlchemyInspectionRepository
from xray_workbench.settings import get_settings
from xray_workbench.vision.detector import Detector
from xray_workbench.vision.fake_detector import FakeDetector
from xray_workbench.vision.yolo_detector import YoloDetector

SessionDependency = Annotated[Session, Depends(get_session)]


@lru_cache
def get_detector() -> Detector:
    settings = get_settings()
    if settings.model_path:
        return YoloDetector(
            settings.model_path,
            model_name=settings.model_name,
            model_version=settings.model_version,
            minimum_confidence=settings.model_min_confidence,
        )
    return FakeDetector(
        model_name=settings.model_name,
        model_version=settings.model_version,
        ready=False,
    )


def get_inspection_service(session: SessionDependency) -> InspectionService:
    settings = get_settings()
    detector = get_detector()
    classifier = RiskClassifier(load_risk_config(settings.risk_config))
    repository = SqlAlchemyInspectionRepository(session)
    return InspectionService(
        detector,
        classifier,
        repository,
        audit_repository=SqlAlchemyAuditRepository(session),
        image_store=LocalImageStore(settings.image_storage_dir),
        image_repository=SqlAlchemyImageReferenceRepository(session),
    )


InspectionServiceDependency = Annotated[InspectionService, Depends(get_inspection_service)]
