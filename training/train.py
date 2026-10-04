"""Run a bounded Ultralytics training job from local weights only."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train from an existing local base-weight file.")
    parser.add_argument(
        "--model",
        required=True,
        help="Local .pt model path; names are not resolved",
    )
    parser.add_argument("--data", default="config/datasets/pidray.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--project", default="training/experiments")
    parser.add_argument("--name", default="pidray-smoke")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = _required_file(args.model, "model")
    data_path = _required_file(args.data, "dataset YAML")
    if args.epochs <= 0 or args.image_size <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs, --image-size, and --batch-size must be greater than zero")
    if args.workers < 0:
        raise SystemExit("--workers cannot be negative")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    options: dict[str, object] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.image_size,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "batch": args.batch_size,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
    }
    if args.device:
        options["device"] = args.device
    model.train(**options)


def _required_file(value: str, description: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise SystemExit(f"Local {description} file does not exist: {path}")
    return path


if __name__ == "__main__":
    main()
