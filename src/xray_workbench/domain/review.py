from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class ReviewDecision(StrEnum):
    CONFIRMED_THREAT = "confirmed_threat"
    FALSE_POSITIVE = "false_positive"
    NEEDS_FURTHER_REVIEW = "needs_further_review"


@dataclass(frozen=True, slots=True)
class OperatorReview:
    decision: ReviewDecision
    notes: str | None = None
    reviewed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
