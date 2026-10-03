import os
from typing import Any

import cv2
import httpx
import numpy as np
import streamlit as st

API_URL = os.getenv("XRAY_API_URL", "http://localhost:8000").rstrip("/")
REVIEWABLE_STATUSES = {"requires_review", "manual_review_required"}


def api_request(method: str, path: str, **kwargs: Any) -> Any:
    with httpx.Client(base_url=API_URL, timeout=30.0) as client:
        response = client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()


def api_binary(path: str) -> bytes:
    with httpx.Client(base_url=API_URL, timeout=30.0) as client:
        response = client.get(path)
        response.raise_for_status()
        return response.content


def annotated_image(payload: bytes, detections: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be decoded")
    height, width = image.shape[:2]
    for detection in detections:
        box = detection["boundingBox"]
        x1 = int(float(box["x"]) * width)
        y1 = int(float(box["y"]) * height)
        x2 = int((float(box["x"]) + float(box["width"])) * width)
        y2 = int((float(box["y"]) + float(box["height"])) * height)
        risk = str(detection["riskLevel"])
        color = (0, 0, 255) if risk == "high" else (0, 180, 255)
        label = f"{detection['className']} {float(detection['confidence']):.0%} {risk}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def submit_review(inspection_id: str, decision: str, notes: str) -> None:
    updated = api_request(
        "POST",
        f"/api/v1/inspections/{inspection_id}/review",
        json={"decision": decision, "notes": notes or None},
    )
    st.session_state["inspection"] = updated
    st.success("Review saved.")


st.set_page_config(page_title="X-Ray Inspection Workbench", layout="wide")
st.title("X-Ray Inspection Workbench")

try:
    health = api_request("GET", "/api/v1/health")
except httpx.HTTPError as error:
    st.error(f"API is unavailable at {API_URL}: {error}")
    st.stop()

if health["status"] == "healthy":
    st.success(f"System healthy · model {health['model']['version']}")
else:
    st.warning("AI model unavailable — uploaded images will require manual review.")

uploaded = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"])
if uploaded is not None and st.button("Start inspection", type="primary"):
    image_bytes = uploaded.getvalue()
    try:
        inspection = api_request(
            "POST",
            "/api/v1/inspections",
            files={"image": (uploaded.name, image_bytes, uploaded.type)},
        )
    except httpx.HTTPError as error:
        st.error(f"Inspection failed: {error}")
    else:
        st.session_state["inspection"] = inspection
        st.session_state["image_bytes"] = image_bytes

inspection = st.session_state.get("inspection")
image_bytes = st.session_state.get("image_bytes")
if inspection is not None:
    st.subheader(f"Inspection {inspection['inspectionId']}")
    image_column, details_column = st.columns([2, 1])
    with image_column:
        if image_bytes is not None:
            st.image(
                annotated_image(image_bytes, inspection["detections"]),
                caption=inspection["imageId"],
                width="stretch",
            )
    with details_column:
        st.metric("Status", inspection["status"])
        st.caption(f"Model version: {inspection['modelVersion'] or 'unavailable'}")
        if inspection["detections"]:
            st.dataframe(inspection["detections"], width="stretch")
        else:
            st.info("No model detections are available.")

    if inspection["status"] in REVIEWABLE_STATUSES:
        st.subheader("Operator review")
        notes = st.text_area("Notes", max_chars=2000)
        confirm, false_positive, further = st.columns(3)
        try:
            if confirm.button("Confirm threat", width="stretch"):
                submit_review(inspection["inspectionId"], "confirmed_threat", notes)
                st.rerun()
            if false_positive.button("False positive / clear", width="stretch"):
                submit_review(inspection["inspectionId"], "false_positive", notes)
                st.rerun()
            if further.button("Further review", width="stretch"):
                submit_review(inspection["inspectionId"], "needs_further_review", notes)
                st.rerun()
        except httpx.HTTPError as error:
            st.error(f"Review could not be saved: {error}")
    elif inspection.get("review"):
        st.success(f"Reviewed: {inspection['review']['decision']}")

    try:
        events = api_request(
            "GET",
            f"/api/v1/inspections/{inspection['inspectionId']}/events",
        )
    except httpx.HTTPError as error:
        st.error(f"Audit trail could not be loaded: {error}")
    else:
        with st.expander("Audit trail", expanded=True):
            for event in events:
                st.markdown(
                    f"**{event['occurredAt']} — {event['eventType']}**  \n{event['message']}"
                )

st.divider()
st.subheader("Recent inspections")
try:
    recent = api_request("GET", "/api/v1/inspections", params={"limit": 25})
except httpx.HTTPError as error:
    st.error(f"History could not be loaded: {error}")
else:
    rows = [
        {
            "inspection": item["inspectionId"],
            "image": item["imageId"],
            "status": item["status"],
            "detections": len(item["detections"]),
            "created": item["createdAt"],
        }
        for item in recent
    ]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
        selected_id = st.selectbox(
            "Open a previous inspection",
            options=[item["inspectionId"] for item in recent],
            format_func=lambda value: next(
                f"{item['imageId']} · {item['status']} · {value}"
                for item in recent
                if item["inspectionId"] == value
            ),
        )
        if st.button("Open selected inspection"):
            try:
                st.session_state["inspection"] = api_request(
                    "GET", f"/api/v1/inspections/{selected_id}"
                )
                st.session_state["image_bytes"] = api_binary(
                    f"/api/v1/inspections/{selected_id}/image"
                )
            except httpx.HTTPError as error:
                st.error(f"Inspection could not be opened: {error}")
            else:
                st.rerun()
    else:
        st.caption("No inspections yet.")
