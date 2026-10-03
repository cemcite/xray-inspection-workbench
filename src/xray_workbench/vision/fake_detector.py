from dataclasses import dataclass, field

from xray_workbench.vision.detector import DetectionCandidate, DetectionResult, ImageArray


@dataclass(slots=True)
class FakeDetector:
    """Deterministic placeholder used by the initial API and tests."""

    candidates: tuple[DetectionCandidate, ...] = field(default_factory=tuple)
    model_name: str = "fake-detector"
    model_version: str = "0.0.0"
    ready: bool = True

    def detect(self, image: ImageArray) -> DetectionResult:
        if not self.ready:
            raise RuntimeError("Detector is unavailable")
        if image.size == 0:
            raise ValueError("Image cannot be empty")
        return DetectionResult(
            detections=self.candidates,
            model_name=self.model_name,
            model_version=self.model_version,
        )
