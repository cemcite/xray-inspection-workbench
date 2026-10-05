"""Prepare a nested 2,000-image train subset and independent 400-image validation subset."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import cv2

from xray_workbench.dataset import convert_coco_split
from xray_workbench.vision.error_analysis import load_yolo_labels

CLASSES = ("gun", "knife", "scissors", "lighter")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=r"C:\Datasets\PIDray\annotations\train.json")
    parser.add_argument("--archive", default=r"C:\Datasets\PIDray\pidray.zip")
    parser.add_argument("--output", default="data/processed/pidray-expanded")
    args = parser.parse_args()
    annotation = Path(args.annotations).resolve()
    archive_path = Path(args.archive).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"Output exists; refusing to mix datasets: {output}")
    source = json.loads(annotation.read_text(encoding="utf-8"))
    source_images = {row["file_name"]: row for row in source["images"]}
    with ZipFile(archive_path) as archive:
        members = set(archive.namelist())
    available = [name for name in source_images if f"pidray/train/{name}" in members]
    print(f"Available annotated training images: {len(available)}/{len(source_images)}", flush=True)

    def convert(split: str, count: int, exclude: list[str]) -> None:
        convert_coco_split(
            annotation,
            None,
            output,
            split=split,
            max_images=count,
            class_names=CLASSES,
            include_backgrounds=True,
            image_archive=archive_path,
            archive_prefix="pidray/train",
            balanced=True,
            background_fraction=0.20,
            include_image_names=available,
            exclude_image_names=exclude,
        )

    convert("train", 2000, [])
    training_names = [path.name for path in (output / "images/train").glob("*.png")]
    convert("val", 400, training_names)

    hashes: dict[str, set[str]] = {}
    summaries = {}
    for split, expected in (("train", 2000), ("val", 400)):
        images = sorted((output / "images" / split).glob("*.png"))
        if len(images) != expected:
            raise RuntimeError(f"Expected {expected} {split} images, found {len(images)}")
        hashes[split] = set()
        counts: Counter[str] = Counter()
        background_count = 0
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"Cannot decode {image_path}")
            row = source_images[image_path.name]
            if image.shape[:2] != (row["height"], row["width"]):
                raise RuntimeError(f"Archive/annotation dimensions disagree: {image_path.name}")
            labels = load_yolo_labels(
                (output / "labels" / split / f"{image_path.stem}.txt").read_text(), len(CLASSES)
            )
            background_count += not labels
            counts.update(CLASSES[label.class_id] for label in labels)
            hashes[split].add(_hash_file(image_path))
        if background_count != round(expected * 0.20) or set(counts) != set(CLASSES):
            raise RuntimeError(f"Incorrect background share or missing class in {split}")
        summaries[split] = {
            "images": len(images),
            "background_images": background_count,
            "class_instances": dict(counts),
            "unique_image_hashes": len(hashes[split]),
        }
    heldout: set[str] = set()
    for split in ("hard", "hidden"):
        directory = Path("data/processed/pidray-hard-hidden/images") / split
        files = sorted(directory.glob("*.png"))
        if len(files) != 160:
            raise RuntimeError(f"Missing fixed 160-image holdout: {directory}")
        heldout.update(_hash_file(path) for path in files)
    overlaps = {
        "train_val": len(hashes["train"] & hashes["val"]),
        "train_hard_hidden": len(hashes["train"] & heldout),
        "val_hard_hidden": len(hashes["val"] & heldout),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Image-content leakage detected: {overlaps}")
    original_names = {
        path.name for path in Path("data/processed/pidray-baseline/images/train").glob("*.png")
    }
    inherited_names = original_names & set(training_names)
    if len(original_names) != 500 or inherited_names != original_names:
        raise RuntimeError("New training subset must retain all 500 original baseline images")
    original_hashes = {
        _hash_file(path)
        for path in Path("data/processed/pidray-baseline/images/train").glob("*.png")
    }
    if not original_hashes <= hashes["train"]:
        raise RuntimeError("Original training image content differs from the expanded subset")
    audit = {
        "source_annotation": str(annotation),
        "source_annotation_sha256": _hash_file(annotation),
        "annotated_train_images": len(source_images),
        "available_annotated_train_images": len(available),
        "splits": summaries,
        "content_overlap_counts": overlaps,
        "original_training_images_retained": len(inherited_names),
        "selection": "Class-balanced order; validation excludes selected train filenames.",
        "background_definition": "No four-class labels; other PIDray classes may exist.",
        "validation_change": "400 held-out train.json images replace easy-split validation.",
    }
    (output / "manifests/audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
