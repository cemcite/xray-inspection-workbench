from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from xray_workbench.domain.detection import BoundingBox

ImageArray = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class DetectionCandidate:
    class_name: str
    confidence: float
    bounding_box: BoundingBox


@dataclass(frozen=True, slots=True)
class DetectionResult:
    detections: tuple[DetectionCandidate, ...] = field(default_factory=tuple)
    model_name: str = "unknown"
    model_version: str = "unknown"
    inference_ms: float = 0.0


class Detector(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def ready(self) -> bool: ...

    def detect(self, image: ImageArray) -> DetectionResult: ...
