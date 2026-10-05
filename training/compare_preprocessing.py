"""Compare raw, CLAHE, and denoise+CLAHE on fixed PIDray evaluation splits."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import cv2
import yaml

from xray_workbench.vision.detector import ImageArray, PreprocessingMode
from xray_workbench.vision.preprocessing import preprocess_image

CLASS_NAMES = ("gun", "knife", "scissors", "lighter")
SPLITS = ("hard", "hidden")
MODES = (
    PreprocessingMode.RAW,
    PreprocessingMode.CLAHE,
    PreprocessingMode.DENOISE_CLAHE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare preprocessing modes using identical PIDray images and labels."
    )
    parser.add_argument(
        "--model",
        default="training/experiments/pidray-baseline-cpu/weights/best.pt",
    )
    parser.add_argument("--data-root", default="data/processed/pidray-hard-hidden")
    parser.add_argument("--output-dir")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    data_root = Path(args.data_root).resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model file does not exist: {model_path}")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be greater than zero")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path("runs")
        / f"preprocessing-comparison-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists; refusing to overwrite: {output_dir}")

    source_sets = {split: _validate_split(data_root, split) for split in SPLITS}
    output_dir.mkdir(parents=True)
    cv2.setNumThreads(1)

    import ultralytics
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(str(model_path))
    records: list[dict[str, Any]] = []
    for split, (images, labels) in source_sets.items():
        for mode in MODES:
            dataset_yaml = _prepare_variant(output_dir, split, mode, images, labels)
            print(f"Evaluating {split}/{mode.value} ({len(images)} images)...", flush=True)
            metrics = model.val(
                data=str(dataset_yaml),
                imgsz=args.image_size,
                device=args.device,
                workers=0,
                plots=False,
                verbose=False,
                project=str(output_dir / "ultralytics"),
                name=f"{split}-{mode.value}",
            )
            record: dict[str, Any] = {
                "split": split,
                "mode": mode.value,
                "images": len(images),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
                "per_class_map50": {
                    CLASS_NAMES[index]: float(metrics.box.maps[index])
                    for index in range(len(CLASS_NAMES))
                },
                "enhancement_ms_per_image_mean": _mean(_transform_times(split, images, mode)),
                "ultralytics_speed_ms_per_image": {
                    key: float(value) for key, value in metrics.speed.items()
                },
            }
            records.append(record)
            print(
                f"  P={record['precision']:.4f} R={record['recall']:.4f} "
                f"mAP50={record['map50']:.4f} mAP50-95={record['map50_95']:.4f} "
                f"enhance={record['enhancement_ms_per_image_mean']:.2f} ms/image",
                flush=True,
            )

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "data_root": str(data_root),
        "image_size": args.image_size,
        "device": args.device,
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "ultralytics": ultralytics.__version__,
        "split_image_counts": {split: len(pair[0]) for split, pair in source_sets.items()},
        "results": records,
        "notes": [
            "Hard and hidden use fixed local 160-image subsets with unchanged labels.",
            "Enhancement latency is a CPU transform measurement; it excludes image decode "
            "and disk I/O.",
            "Ultralytics speed values are separate from enhancement latency.",
            "Metrics are subset diagnostics, not full PIDray benchmark results.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "report.csv", records)
    print(f"Saved report: {output_dir / 'report.json'}", flush=True)


def _validate_split(data_root: Path, split: str) -> tuple[list[Path], list[Path]]:
    image_dir = data_root / "images" / split
    label_dir = data_root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(f"Missing image or label directory for split '{split}' under {data_root}")
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if len(images) != 160:
        raise SystemExit(f"Expected the fixed 160-image '{split}' subset, found {len(images)}")
    labels = [label_dir / f"{image.stem}.txt" for image in images]
    missing = [path for path in labels if not path.is_file()]
    if not images or missing:
        raise SystemExit(
            f"Split '{split}' has {len(images)} images and {len(missing)} missing labels"
        )
    return images, labels


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _prepare_variant(
    output_dir: Path,
    split: str,
    mode: PreprocessingMode,
    images: list[Path],
    labels: list[Path],
) -> Path:
    if mode is PreprocessingMode.RAW:
        dataset_root = images[0].parents[2]
        image_dir = images[0].parent
        label_dir = labels[0].parent
    else:
        dataset_root = output_dir / "datasets" / split / mode.value
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image_path, label_path in zip(images, labels, strict=True):
            image = cast(ImageArray, cv2.imread(str(image_path), cv2.IMREAD_COLOR))
            if image is None:
                raise RuntimeError(f"OpenCV could not decode image: {image_path}")
            started = perf_counter()
            processed = preprocess_image(image, mode)
            elapsed_ms = (perf_counter() - started) * 1000.0
            if not cv2.imwrite(str(image_dir / image_path.name), processed):
                raise RuntimeError(f"OpenCV could not write processed image: {image_path.name}")
            shutil.copy2(label_path, label_dir / label_path.name)
            _TRANSFORM_TIMES.setdefault((split, mode), []).append(elapsed_ms)

    yaml_path = output_dir / f"{split}-{mode.value}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": dataset_root.as_posix(),
                "train": f"images/{split}",
                "val": f"images/{split}",
                "names": dict(enumerate(CLASS_NAMES)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path


_TRANSFORM_TIMES: dict[tuple[str, PreprocessingMode], list[float]] = {}


def _transform_times(
    split: str, images: list[Path], mode: PreprocessingMode
) -> list[float]:
    if mode is PreprocessingMode.RAW:
        return [0.0] * len(images)
    return _TRANSFORM_TIMES[(split, mode)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = (
        "split",
        "mode",
        "images",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "enhancement_ms_per_image_mean",
        "ultralytics_speed_ms_per_image",
        "per_class_map50",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "ultralytics_speed_ms_per_image": json.dumps(
                        record["ultralytics_speed_ms_per_image"], sort_keys=True
                    ),
                    "per_class_map50": json.dumps(record["per_class_map50"], sort_keys=True),
                }
            )


if __name__ == "__main__":
    main()
