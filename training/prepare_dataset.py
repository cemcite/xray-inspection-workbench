"""Convert a locally obtained COCO-format PIDray split to YOLO format."""

import argparse
import json
from dataclasses import asdict

from xray_workbench.dataset import convert_coco_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one local COCO split without downloading any dataset files."
    )
    parser.add_argument("--annotations", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images")
    source.add_argument("--image-archive")
    parser.add_argument("--archive-prefix", default="")
    parser.add_argument("--output", default="data/processed/pidray")
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["gun", "knife", "scissors", "lighter"],
    )
    parser.add_argument("--include-backgrounds", action="store_true")
    parser.add_argument("--background-fraction", type=float, default=0.0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_coco_split(
        args.annotations,
        args.images,
        args.output,
        split=args.split,
        class_names=args.classes,
        include_backgrounds=args.include_backgrounds,
        max_images=args.max_images,
        overwrite=args.overwrite,
        image_archive=args.image_archive,
        archive_prefix=args.archive_prefix,
        balanced=args.balanced,
        background_fraction=args.background_fraction,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
