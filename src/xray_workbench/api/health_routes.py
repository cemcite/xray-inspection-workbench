from fastapi import APIRouter
from sqlalchemy import text

from xray_workbench.api.dependencies import SessionDependency, get_detector
from xray_workbench.api.schemas import HealthResponse, ModelHealth

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(session: SessionDependency) -> HealthResponse:
    database_status = "ready"
    try:
        session.execute(text("SELECT 1"))
    except Exception:  # Health boundary intentionally converts failures to status.
        database_status = "unavailable"

    detector = get_detector()
    model_status = "ready" if detector.ready else "unavailable"
    overall_status = "healthy" if database_status == "ready" and detector.ready else "degraded"
    return HealthResponse(
        status=overall_status,
        model=ModelHealth(
            status=model_status,
            name=detector.model_name,
            version=detector.model_version,
        ),
        database=database_status,
    )
