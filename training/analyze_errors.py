"""Generate fixed-threshold detection diagnostics and a local review gallery."""

import argparse
import csv
import hashlib
import html
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import cv2

from xray_workbench.vision.error_analysis import LabelledBox, load_yolo_labels, match_boxes

CLASS_NAMES = ("gun", "knife", "scissors", "lighter")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="training/experiments/pidray-baseline-cpu/weights/best.pt"
    )
    parser.add_argument("--data-root", default="data/processed/pidray-hard-hidden")
    parser.add_argument("--output-dir")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or not 0 < args.iou <= 1 or args.image_size <= 0:
        raise SystemExit("Invalid confidence, IoU, or image size")
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise SystemExit(f"Missing local model: {model_path}")
    data_root = Path(args.data_root).resolve()
    output = Path(args.output_dir or f"runs/error-analysis-{datetime.now(UTC):%Y%m%dT%H%M%SZ}")
    output = output.resolve()
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")

    # Validate every annotation before creating output or loading the model.
    datasets: dict[str, list[tuple[Path, list[LabelledBox]]]] = {}
    data_hash = hashlib.sha256()
    for split in ("hard", "hidden"):
        image_dir = data_root / "images" / split
        images = sorted(path for path in image_dir.glob("*") if path.suffix.lower() == ".png")
        if not images:
            raise SystemExit(f"No PNG images found: {image_dir}")
        datasets[split] = []
        for image in images:
            label = data_root / "labels" / split / f"{image.stem}.txt"
            text = label.read_text(encoding="utf-8")
            datasets[split].append((image, load_yolo_labels(text, len(CLASS_NAMES))))
            for path in (image, label):
                data_hash.update(path.relative_to(data_root).as_posix().encode())
                with path.open("rb") as stream:
                    data_hash.update(hashlib.file_digest(stream, "sha256").digest())

    import ultralytics
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(str(model_path))
    if tuple(model.names[i] for i in range(len(model.names))) != CLASS_NAMES:
        raise SystemExit(f"Model classes do not match PIDray subset: {model.names}")
    output.mkdir(parents=True)
    cases: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for split, samples in datasets.items():
        print(f"Analyzing {split}: {len(samples)} images...", flush=True)
        counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES}
        backgrounds = 0
        background_alarms = 0
        started = perf_counter()
        results = model.predict(
            source=[str(image) for image, _ in samples],
            stream=True,
            conf=args.confidence,
            imgsz=args.image_size,
            device=args.device,
            verbose=False,
        )
        for (image, truth), raw_result in zip(samples, results, strict=True):
            result = cast(Any, raw_result)
            if Path(result.path).resolve() != image:
                raise RuntimeError("Prediction order does not match source images")
            predictions: list[LabelledBox] = []
            if result.boxes is not None:
                for box, class_id, confidence in zip(
                    result.boxes.xyxyn.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    strict=True,
                ):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    predictions.append(
                        LabelledBox(int(class_id), (x1, y1, x2, y2), float(confidence))
                    )
            matched = match_boxes(
                predictions, truth, minimum_confidence=args.confidence, minimum_iou=args.iou
            )
            for _, target in matched.matches:
                counts[CLASS_NAMES[truth[target].class_id]]["tp"] += 1
            for index in matched.false_positives:
                counts[CLASS_NAMES[predictions[index].class_id]]["fp"] += 1
            for index in matched.false_negatives:
                counts[CLASS_NAMES[truth[index].class_id]]["fn"] += 1
            backgrounds += not truth
            background_alarms += not truth and bool(matched.false_positives)
            cases.append(
                {
                    "split": split,
                    "image": str(image),
                    "ground_truth": [asdict(b) for b in truth],
                    "predictions": [asdict(b) for b in predictions],
                    **asdict(matched),
                    "tp": len(matched.matches),
                    "fp": len(matched.false_positives),
                    "fn": len(matched.false_negatives),
                }
            )
        per_class = {name: _scores(values) for name, values in counts.items()}
        total = {key: sum(values[key] for values in counts.values()) for key in ("tp", "fp", "fn")}
        summaries[split] = {
            "images": len(samples),
            **_scores(total),
            "per_class": per_class,
            "background_images": backgrounds,
            "background_images_with_false_alarms": background_alarms,
            "pipeline_wall_seconds": perf_counter() - started,
        }
        print(json.dumps({split: summaries[split]}, indent=2), flush=True)

    with model_path.open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": str(model_path),
        "model_sha256": model_hash,
        "dataset_sha256": data_hash.hexdigest(),
        "confidence_threshold": args.confidence,
        "iou_threshold": args.iou,
        "image_size": args.image_size,
        "device": args.device,
        "preprocessing": "raw",
        "ultralytics": ultralytics.__version__,
        "summaries": summaries,
        "cases": cases,
        "method": "Confidence-ordered greedy, class-aware, one-to-one IoU matching.",
        "limitations": [
            "Fixed-threshold precision/recall/F1 differ from Ultralytics AP validation metrics.",
            "Background means no annotations for the four selected classes, not a safety judgment.",
            "Small subsets; runtime includes decoding and startup, not a latency benchmark.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("split", "image", "tp", "fp", "fn"), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(cases)
    _gallery(output, cases)
    print(f"Review gallery: {output / 'gallery.html'}", flush=True)


def _scores(counts: dict[str, int]) -> dict[str, int | float | None]:
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {
        **counts,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def _gallery(output: Path, cases: list[dict[str, Any]]) -> None:
    previews = output / "previews"
    previews.mkdir()
    cards: list[str] = []
    for split in ("hard", "hidden"):
        subset = [case for case in cases if case["split"] == split]
        selected: dict[str, dict[str, Any]] = {}
        for key in ("fn", "fp"):
            ranked = sorted(subset, key=lambda case: (-case[key], case["image"]))
            for case in [item for item in ranked if item[key] > 0][:6]:
                selected[case["image"]] = case
        for case in selected.values():
            image_path = Path(case["image"])
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Cannot decode preview: {image_path}")
            height, width = image.shape[:2]
            for label_key, color, prefix in (
                ("ground_truth", (0, 200, 0), "GT"),
                ("predictions", (0, 140, 255), "Pred"),
            ):
                for box in case[label_key]:
                    x1, y1, x2, y2 = box["xyxy"]
                    start = (round(x1 * width), round(y1 * height))
                    end = (round(x2 * width), round(y2 * height))
                    cv2.rectangle(image, start, end, color, 2)
                    label = f"{prefix} {CLASS_NAMES[box['class_id']]}"
                    if prefix == "Pred":
                        label += f" {box['confidence']:.2f}"
                    cv2.putText(
                        image,
                        label,
                        (max(0, start[0]), max(15, start[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )
            filename = f"{split}-{image_path.stem}.png"
            if not cv2.imwrite(str(previews / filename), image):
                raise RuntimeError(f"Cannot write preview: {filename}")
            cards.append(
                f"<article><h2>{html.escape(split + '/' + image_path.name)}</h2>"
                f"<p>TP {case['tp']} · FP {case['fp']} · FN {case['fn']}</p>"
                f'<img src="previews/{html.escape(filename)}" alt="Detection review"></article>'
            )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>PIDray error review</title>'
        "<style>body{font-family:system-ui;margin:24px;background:#141820;color:#eef}"
        "article{padding:16px;border:1px solid #556;margin:16px 0}img{max-width:100%;height:auto}"
        "h2{font-size:18px}</style><h1>PIDray fixed-threshold error review</h1>"
        "<p>Green: ground truth. Orange: predictions. Each split shows up to six cases with the "
        "most missed objects and six with the most false positives (duplicates removed).</p>"
        "<p>Other PIDray classes are excluded from annotations. Subset diagnostics only.</p>"
        + "".join(cards)
        + "</html>"
    )
    (output / "gallery.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
