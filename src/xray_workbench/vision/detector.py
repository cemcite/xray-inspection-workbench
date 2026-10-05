from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np
import numpy.typing as npt

from xray_workbench.domain.detection import BoundingBox

ImageArray = npt.NDArray[np.uint8]


class PreprocessingMode(StrEnum):
    RAW = "raw"
    CLAHE = "clahe"
    DENOISE_CLAHE = "denoise_clahe"


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
    preprocessing_ms: float = 0.0
    postprocessing_ms: float = 0.0
    preprocessing_mode: PreprocessingMode = PreprocessingMode.RAW


class Detector(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def ready(self) -> bool: ...

    def detect(self, image: ImageArray) -> DetectionResult: ...
