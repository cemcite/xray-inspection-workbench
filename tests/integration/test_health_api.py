import cv2
import numpy as np
from fastapi.testclient import TestClient

from xray_workbench.api.main import app


def test_health_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "model": {
            "status": "unavailable",
            "name": "xray-detector",
            "version": "unavailable",
        },
        "database": "ready",
    }


def test_upload_fails_safe_when_model_is_unavailable() -> None:
    encoded, image = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert encoded is True

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inspections",
            files={"image": ("sample.png", image.tobytes(), "image/png")},
        )
        inspection_id = response.json()["inspectionId"]
        review = client.post(
            f"/api/v1/inspections/{inspection_id}/review",
            json={"decision": "false_positive", "notes": "Manual inspection completed."},
        )
        stored_image = client.get(f"/api/v1/inspections/{inspection_id}/image")
        events = client.get(f"/api/v1/inspections/{inspection_id}/events")
        recent = client.get("/api/v1/inspections", params={"limit": 25})

    assert response.status_code == 201
    assert response.json()["status"] == "manual_review_required"
    assert review.status_code == 200
    assert review.json()["status"] == "cleared"
    assert review.json()["review"]["decision"] == "false_positive"
    assert stored_image.status_code == 200
    assert stored_image.headers["content-type"] == "image/png"
    assert stored_image.content == image.tobytes()
    assert events.status_code == 200
    assert [item["eventType"] for item in events.json()] == [
        "image_received",
        "processing_started",
        "model_unavailable",
        "review_recorded",
    ]
    assert recent.status_code == 200
    assert inspection_id in {item["inspectionId"] for item in recent.json()}
