from pathlib import Path
from typing import Annotated, cast
from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status

from xray_workbench.api.dependencies import InspectionServiceDependency
from xray_workbench.api.schemas import AuditEventResponse, InspectionResponse, ReviewRequest
from xray_workbench.application.inspection_service import (
    InspectionImageNotFoundError,
    InspectionNotFoundError,
)
from xray_workbench.domain.review import OperatorReview
from xray_workbench.vision.detector import ImageArray

router = APIRouter(prefix="/inspections", tags=["inspections"])


@router.post("", response_model=InspectionResponse, status_code=status.HTTP_201_CREATED)
async def create_inspection(
    service: InspectionServiceDependency,
    image: Annotated[UploadFile, File()],
) -> InspectionResponse:
    payload = await image.read()
    encoded = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image")
    image_id = Path(image.filename or "uploaded-image").name
    inspection = service.inspect(
        image_id=image_id,
        image=cast(ImageArray, decoded),
        image_payload=payload,
        content_type=_image_content_type(payload),
    )
    return InspectionResponse.from_domain(inspection)


@router.get("", response_model=list[InspectionResponse])
def list_inspections(
    service: InspectionServiceDependency,
    limit: int = 50,
) -> list[InspectionResponse]:
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    return [InspectionResponse.from_domain(item) for item in service.list_recent(limit)]


@router.get("/{inspection_id}", response_model=InspectionResponse)
def get_inspection(
    inspection_id: UUID,
    service: InspectionServiceDependency,
) -> InspectionResponse:
    try:
        inspection = service.get(inspection_id)
    except InspectionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Inspection not found") from error
    return InspectionResponse.from_domain(inspection)


@router.get("/{inspection_id}/image")
def get_inspection_image(
    inspection_id: UUID,
    service: InspectionServiceDependency,
) -> Response:
    try:
        stored, payload = service.get_image(inspection_id)
    except InspectionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Inspection not found") from error
    except InspectionImageNotFoundError as error:
        raise HTTPException(status_code=404, detail="Inspection image not found") from error
    return Response(content=payload, media_type=stored.content_type)


@router.get("/{inspection_id}/events", response_model=list[AuditEventResponse])
def get_inspection_events(
    inspection_id: UUID,
    service: InspectionServiceDependency,
) -> list[AuditEventResponse]:
    try:
        events = service.list_events(inspection_id)
    except InspectionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Inspection not found") from error
    return [AuditEventResponse.from_domain(event) for event in events]


@router.post("/{inspection_id}/review", response_model=InspectionResponse)
def review_inspection(
    inspection_id: UUID,
    request: ReviewRequest,
    service: InspectionServiceDependency,
) -> InspectionResponse:
    try:
        inspection = service.review(
            inspection_id,
            OperatorReview(decision=request.decision, notes=request.notes),
        )
    except InspectionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Inspection not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return InspectionResponse.from_domain(inspection)


def _image_content_type(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported")
