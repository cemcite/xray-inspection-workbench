from types import SimpleNamespace

import pytest

from xray_workbench.vision.error_analysis import (
    LabelledBox,
    intersection_over_union,
    load_yolo_labels,
    match_boxes,
    per_class_average_precision,
)


def test_duplicate_predictions_do_not_double_count_a_true_positive() -> None:
    truth = [LabelledBox(0, (0, 0, 1, 1))]
    predictions = [LabelledBox(0, (0, 0, 1, 1), 0.4), LabelledBox(0, (0, 0, 1, 1), 0.9)]
    result = match_boxes(predictions, truth)
    assert result.matches == ((1, 0),)
    assert result.false_positives == (0,)
    assert result.false_negatives == ()


def test_wrong_class_is_both_false_positive_and_false_negative() -> None:
    result = match_boxes([LabelledBox(1, (0, 0, 1, 1))], [LabelledBox(0, (0, 0, 1, 1))])
    assert result.matches == ()
    assert result.false_positives == (0,)
    assert result.false_negatives == (0,)


def test_low_confidence_is_filtered_and_background_false_alarms_count() -> None:
    predictions = [LabelledBox(0, (0, 0, 1, 1), 0.09), LabelledBox(0, (0, 0, 1, 1), 0.10)]
    assert match_boxes(predictions, []).false_positives == (1,)
    assert match_boxes(predictions[:1], [LabelledBox(0, (0, 0, 1, 1))]).false_negatives == (0,)


def test_overlapping_boxes_below_iou_threshold_are_not_matched() -> None:
    left, right = LabelledBox(0, (0, 0, 1, 1)), LabelledBox(0, (0.5, 0, 1.5, 1))
    assert intersection_over_union(left, right) == pytest.approx(1 / 3)
    assert match_boxes([left], [right]).matches == ()


def test_yolo_labels_convert_centers_and_allow_empty_backgrounds() -> None:
    assert load_yolo_labels("", 4) == []
    assert load_yolo_labels("1 0.5 0.5 0.2 0.4", 4)[0].xyxy == pytest.approx((0.4, 0.3, 0.6, 0.7))


@pytest.mark.parametrize("text", ["4 0.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1", "0 0.5"])
def test_invalid_labels_fail_loudly(text: str) -> None:
    with pytest.raises(ValueError):
        load_yolo_labels(text, 4)


def test_per_class_ap_uses_ground_truth_class_ids_and_correct_iou_metric() -> None:
    metrics = SimpleNamespace(ap_class_index=[3, 0], ap50=[0.8, 0.6], ap=[0.4, 0.3])
    result = per_class_average_precision(metrics, ("gun", "knife", "scissors", "lighter"))
    assert result["per_class_map50"] == {
        "gun": 0.6,
        "knife": None,
        "scissors": None,
        "lighter": 0.8,
    }
    assert result["per_class_map50_95"]["lighter"] == 0.4
