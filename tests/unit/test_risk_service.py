from pathlib import Path

import pytest

from xray_workbench.application.risk_service import (
    RiskClassifier,
    RiskConfig,
    RiskThresholds,
    load_risk_config,
)
from xray_workbench.domain.risk import RiskLevel


@pytest.fixture
def classifier() -> RiskClassifier:
    return RiskClassifier(
        RiskConfig(
            default=RiskThresholds(review_threshold=0.5, high_threshold=0.8),
            classes={"gun": RiskThresholds(review_threshold=0.3, high_threshold=0.65)},
        )
    )


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.29, RiskLevel.LOW), (0.30, RiskLevel.REVIEW), (0.65, RiskLevel.HIGH)],
)
def test_class_specific_threshold_boundaries(
    classifier: RiskClassifier,
    confidence: float,
    expected: RiskLevel,
) -> None:
    assert classifier.classify("gun", confidence) is expected


def test_unknown_class_uses_default_thresholds(classifier: RiskClassifier) -> None:
    assert classifier.classify("unknown", 0.6) is RiskLevel.REVIEW


def test_invalid_threshold_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="review <= high"):
        RiskThresholds(review_threshold=0.9, high_threshold=0.8)


def test_yaml_config_loader(tmp_path: Path) -> None:
    path = tmp_path / "risk.yaml"
    path.write_text(
        "default:\n  review_threshold: 0.4\n  high_threshold: 0.8\nclasses: {}\n",
        encoding="utf-8",
    )

    config = load_risk_config(path)

    assert config.default == RiskThresholds(review_threshold=0.4, high_threshold=0.8)
