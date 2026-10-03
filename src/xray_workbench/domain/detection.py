from dataclasses import dataclass

from xray_workbench.domain.risk import RiskLevel


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Resolution-independent XYWH box with values normalized to [0, 1]."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Bounding-box values must be normalized to [0, 1]")
        if self.width == 0.0 or self.height == 0.0:
            raise ValueError("Bounding-box width and height must be greater than zero")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("Bounding box must stay within image bounds")


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    risk_level: RiskLevel

    def __post_init__(self) -> None:
        if not self.class_name.strip():
            raise ValueError("Detection class_name cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be in [0, 1]")
