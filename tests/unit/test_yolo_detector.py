from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xray_workbench.vision.yolo_detector import ModelUnavailableError, YoloDetector


class StubModel:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    def predict(self, **kwargs: object) -> list[object]:
        self.arguments = kwargs
        boxes = SimpleNamespace(
            xywhn=np.array([[0.50, 0.40, 0.20, 0.30]], dtype=np.float32),
            conf=np.array([0.91], dtype=np.float32),
            cls=np.array([1], dtype=np.float32),
        )
        return [
            SimpleNamespace(
                boxes=boxes,
                names={0: "gun", 1: "knife"},
                speed={"inference": 12.5},
            )
        ]


def test_maps_ultralytics_center_xywh_to_domain_top_left(tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.touch()
    model = StubModel()
    detector = YoloDetector(
        weights,
        model_name="xray-detector",
        model_version="0.1.0",
        minimum_confidence=0.10,
        model_loader=lambda _: model,
    )

    result = detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))

    detection = result.detections[0]
    assert detection.class_name == "knife"
    assert detection.confidence == pytest.approx(0.91)
    assert detection.bounding_box.x == pytest.approx(0.40)
    assert detection.bounding_box.y == pytest.approx(0.25)
    assert detection.bounding_box.width == pytest.approx(0.20)
    assert detection.bounding_box.height == pytest.approx(0.30)
    assert result.inference_ms == 12.5
    assert model.arguments["conf"] == 0.10
    assert "device" not in model.arguments


def test_missing_weights_do_not_trigger_loader(tmp_path: Path) -> None:
    loader_called = False

    def loader(_: str) -> StubModel:
        nonlocal loader_called
        loader_called = True
        return StubModel()

    detector = YoloDetector(
        tmp_path / "missing.pt",
        model_name="xray-detector",
        model_version="0.1.0",
        model_loader=loader,
    )

    assert detector.ready is False
    assert loader_called is False
    with pytest.raises(ModelUnavailableError, match="do not exist"):
        detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))
