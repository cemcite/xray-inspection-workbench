from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from xray_workbench.domain.risk import RiskLevel


@dataclass(frozen=True, slots=True)
class RiskThresholds:
    review_threshold: float
    high_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.review_threshold <= self.high_threshold <= 1.0:
            raise ValueError("Thresholds must satisfy 0 <= review <= high <= 1")


@dataclass(frozen=True, slots=True)
class RiskConfig:
    default: RiskThresholds
    classes: dict[str, RiskThresholds]


class RiskClassifier:
    def __init__(self, config: RiskConfig) -> None:
        self._config = config

    def classify(self, class_name: str, confidence: float) -> RiskLevel:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be in [0, 1]")
        thresholds = self._config.classes.get(class_name, self._config.default)
        if confidence >= thresholds.high_threshold:
            return RiskLevel.HIGH
        if confidence >= thresholds.review_threshold:
            return RiskLevel.REVIEW
        return RiskLevel.LOW


def load_risk_config(path: str | Path) -> RiskConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Risk configuration root must be a mapping")
    default = _parse_thresholds(raw.get("default"), "default")
    raw_classes = raw.get("classes", {})
    if not isinstance(raw_classes, dict):
        raise ValueError("Risk configuration 'classes' must be a mapping")
    classes = {
        str(class_name): _parse_thresholds(value, f"classes.{class_name}")
        for class_name, value in raw_classes.items()
    }
    return RiskConfig(default=default, classes=classes)


def _parse_thresholds(value: Any, location: str) -> RiskThresholds:
    if not isinstance(value, dict):
        raise ValueError(f"Risk configuration '{location}' must be a mapping")
    try:
        return RiskThresholds(
            review_threshold=float(value["review_threshold"]),
            high_threshold=float(value["high_threshold"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid thresholds at '{location}'") from error
