"""Class-aware, one-to-one matching at a fixed confidence and IoU threshold."""

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class LabelledBox:
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.xyxy
        if self.class_id < 0 or not all(isfinite(v) for v in self.xyxy):
            raise ValueError("Class IDs must be nonnegative and coordinates finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Box width and height must be positive")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class ImageMatches:
    # Each pair contains (prediction index, ground-truth index).
    matches: tuple[tuple[int, int], ...]
    false_positives: tuple[int, ...]
    false_negatives: tuple[int, ...]


def intersection_over_union(left: LabelledBox, right: LabelledBox) -> float:
    a, b = left.xyxy, right.xyxy
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def match_boxes(
    predictions: list[LabelledBox],
    ground_truth: list[LabelledBox],
    *,
    minimum_confidence: float = 0.10,
    minimum_iou: float = 0.50,
) -> ImageMatches:
    """Match high-confidence predictions first, within the same class.

    Unmatched predictions are FP; unmatched labels are FN. A wrong class thus
    contributes one FP and one FN. Duplicate detections cannot claim the same
    ground-truth object. This diagnostic rule is separate from AP integration.
    """
    if not 0 <= minimum_confidence <= 1 or not 0 < minimum_iou <= 1:
        raise ValueError("Confidence must be in [0, 1] and IoU in (0, 1]")
    remaining = set(range(len(ground_truth)))
    matches: list[tuple[int, int]] = []
    false_positives: list[int] = []
    for index in sorted(range(len(predictions)), key=lambda i: (-predictions[i].confidence, i)):
        prediction = predictions[index]
        if prediction.confidence < minimum_confidence:
            continue
        candidates = [
            (intersection_over_union(prediction, ground_truth[target]), target)
            for target in sorted(remaining)
            if prediction.class_id == ground_truth[target].class_id
        ]
        best = max(candidates, default=(0.0, -1), key=lambda pair: pair[0])
        if best[0] >= minimum_iou:
            remaining.remove(best[1])
            matches.append((index, best[1]))
        else:
            false_positives.append(index)
    return ImageMatches(tuple(matches), tuple(false_positives), tuple(sorted(remaining)))


def load_yolo_labels(text: str, class_count: int) -> list[LabelledBox]:
    labels: list[LabelledBox] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Label line {line_number} must contain class and four coordinates")
        class_id = int(parts[0])
        cx, cy, width, height = (float(part) for part in parts[1:])
        if not 0 <= class_id < class_count or not all(
            isfinite(v) and 0 <= v <= 1 for v in (cx, cy, width, height)
        ):
            raise ValueError(f"Invalid class or normalized coordinates at label line {line_number}")
        labels.append(
            LabelledBox(
                class_id, (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
            )
        )
    return labels


def per_class_average_precision(
    box: Any, class_names: tuple[str, ...]
) -> dict[str, dict[str, float | None]]:
    """Map AP arrays by ground-truth class IDs; absent classes remain unmeasured."""
    ap50: dict[str, float | None] = dict.fromkeys(class_names)
    ap95: dict[str, float | None] = dict.fromkeys(class_names)
    for class_id, score50, score95 in zip(box.ap_class_index, box.ap50, box.ap, strict=True):
        name = class_names[int(class_id)]
        ap50[name] = float(score50)
        ap95[name] = float(score95)
    return {"per_class_map50": ap50, "per_class_map50_95": ap95}
