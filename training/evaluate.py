"""Evaluate an existing local Ultralytics model without downloading weights."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local model on a local dataset.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="config/datasets/pidray.yaml")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = _required_file(args.model, "model")
    data_path = _required_file(args.data, "dataset YAML")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be greater than zero")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    options: dict[str, object] = {"data": str(data_path), "imgsz": args.image_size}
    if args.device:
        options["device"] = args.device
    model.val(**options)


def _required_file(value: str, description: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise SystemExit(f"Local {description} file does not exist: {path}")
    return path


if __name__ == "__main__":
    main()
