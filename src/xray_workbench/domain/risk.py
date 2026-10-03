from enum import StrEnum


class RiskLevel(StrEnum):
    """Operational policy result, separate from model confidence."""

    LOW = "low"
    REVIEW = "review"
    HIGH = "high"
