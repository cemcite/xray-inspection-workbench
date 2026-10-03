"""Ultralytics adapter with lazy third-party imports and no weight downloads."""

from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

import numpy as np

from xray_workbench.domain.detection import BoundingBox
from xray_workbench.vision.detector import DetectionCandidate, DetectionResult, ImageArray


class ModelUnavailableError(RuntimeError):
    pass


class PredictModel(Protocol):
    def predict(self, **kwargs: object) -> Sequence[Any]: ...


ModelLoader = Callable[[str], PredictModel]


class YoloDetector:
    """Map Ultralytics results into the local detector contract.

    The weights path must already exist. This adapter never asks Ultralytics to
    resolve a model name, which prevents an accidental network download.
    """

    def __init__(
        self,
        weights_path: str | Path,
        *,
        model_name: str,
        model_version: str,
        minimum_confidence: float = 0.10,
        device: str | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        self._weights_path = Path(weights_path)
        self._model_name = model_name
        self._model_version = model_version
        self._minimum_confidence = minimum_confidence
        self._device = device
        self._model: PredictModel | None = None
        self._load_error: str | None = None

        if not self._weights_path.is_file():
            self._load_error = f"Model weights do not exist: {self._weights_path}"
            return

        loader = model_loader or _load_ultralytics_model
        try:
            self._model = loader(str(self._weights_path))
        except Exception as error:
            self._load_error = str(error)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def detect(self, image: ImageArray) -> DetectionResult:
        if self._model is None:
            raise ModelUnavailableError(self._load_error or "Model is unavailable")
        if image.size == 0:
            raise ValueError("Image cannot be empty")

        arguments: dict[str, object] = {
            "source": image,
            "conf": self._minimum_confidence,
            "verbose": False,
        }
        if self._device is not None:
            arguments["device"] = self._device

        started = perf_counter()
        results = self._model.predict(**arguments)
        elapsed_ms = (perf_counter() - started) * 1000.0
        if len(results) != 1:
            raise RuntimeError(f"Expected one inference result, received {len(results)}")

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return DetectionResult(
                model_name=self.model_name,
                model_version=self.model_version,
                inference_ms=_inference_time(result, elapsed_ms),
            )

        xywhn = _to_numpy(boxes.xywhn)
        confidences = _to_numpy(boxes.conf)
        class_ids = _to_numpy(boxes.cls)
        candidates = tuple(
            DetectionCandidate(
                class_name=str(result.names[int(class_id)]),
                confidence=float(confidence),
                bounding_box=_center_xywh_to_top_left(box),
            )
            for box, confidence, class_id in zip(
                xywhn,
                confidences,
                class_ids,
                strict=True,
            )
        )
        return DetectionResult(
            detections=candidates,
            model_name=self.model_name,
            model_version=self.model_version,
            inference_ms=_inference_time(result, elapsed_ms),
        )


def _load_ultralytics_model(weights_path: str) -> PredictModel:
    from ultralytics import YOLO

    return cast(PredictModel, YOLO(weights_path))


def _to_numpy(value: Any) -> np.ndarray[Any, Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _center_xywh_to_top_left(box: Any) -> BoundingBox:
    center_x, center_y, width, height = (float(item) for item in box)
    x = max(0.0, min(1.0, center_x - width / 2.0))
    y = max(0.0, min(1.0, center_y - height / 2.0))
    width = max(0.0, min(width, 1.0 - x))
    height = max(0.0, min(height, 1.0 - y))
    return BoundingBox(x=x, y=y, width=width, height=height)


def _inference_time(result: Any, fallback_ms: float) -> float:
    speed = getattr(result, "speed", None)
    if isinstance(speed, dict):
        value = speed.get("inference")
        if isinstance(value, (int, float)):
            return float(value)
    return fallback_ms
