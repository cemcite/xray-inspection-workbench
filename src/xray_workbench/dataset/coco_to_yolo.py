import json
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CocoAnnotation:
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ConversionSummary:
    split: str
    image_count: int
    annotation_count: int
    skipped_annotation_count: int
    class_names: tuple[str, ...]


def convert_coco_split(
    annotation_file: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str,
    class_names: Sequence[str],
    include_backgrounds: bool = False,
    max_images: int | None = None,
    overwrite: bool = False,
) -> ConversionSummary:
    """Convert one COCO detection split to YOLO labels and copied images."""

    if not split or Path(split).name != split:
        raise ValueError("split must be a single non-empty path segment")
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be non-empty and unique")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be greater than zero")

    annotation_path = Path(annotation_file)
    source_images = Path(images_dir)
    destination = Path(output_dir)
    raw = _read_json_object(annotation_path)

    categories = _category_map(raw.get("categories"))
    requested = {name.casefold(): index for index, name in enumerate(class_names)}
    category_to_class = {
        category_id: requested[name.casefold()]
        for category_id, name in categories.items()
        if name.casefold() in requested
    }
    missing = sorted(set(requested) - {name.casefold() for name in categories.values()})
    if missing:
        raise ValueError(f"Requested categories are absent from COCO metadata: {missing}")

    images = _images(raw.get("images"))
    annotations = _annotations(raw.get("annotations"))
    annotations_by_image: dict[int, list[CocoAnnotation]] = defaultdict(list)
    for annotation in annotations:
        if annotation.category_id in category_to_class:
            annotations_by_image[annotation.image_id].append(annotation)

    selected_images = [
        image
        for image in sorted(images, key=lambda item: item.file_name)
        if include_backgrounds or annotations_by_image[image.id]
    ]
    if max_images is not None:
        selected_images = selected_images[:max_images]

    image_output = destination / "images" / split
    label_output = destination / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    annotation_count = 0
    skipped_count = 0
    for image in selected_images:
        source = source_images / image.file_name
        target = image_output / Path(image.file_name).name
        if not source.is_file():
            raise FileNotFoundError(f"COCO image is missing: {source}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {target}")
        shutil.copy2(source, target)

        lines: list[str] = []
        for annotation in annotations_by_image[image.id]:
            normalized = _normalize_bbox(annotation.bbox, image.width, image.height)
            if normalized is None:
                skipped_count += 1
                continue
            class_id = category_to_class[annotation.category_id]
            lines.append(f"{class_id} " + " ".join(f"{value:.8f}" for value in normalized))
            annotation_count += 1
        (label_output / f"{target.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    summary = ConversionSummary(
        split=split,
        image_count=len(selected_images),
        annotation_count=annotation_count,
        skipped_annotation_count=skipped_count,
        class_names=tuple(class_names),
    )
    manifest_dir = destination / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{split}.json").write_text(
        json.dumps(asdict(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value: Any = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("COCO annotation root must be an object")
    return value


def _category_map(value: object) -> dict[int, str]:
    rows = _object_list(value, "categories")
    return {_int(row, "id"): _str(row, "name") for row in rows}


def _images(value: object) -> list[CocoImage]:
    return [
        CocoImage(
            id=_int(row, "id"),
            file_name=_str(row, "file_name"),
            width=_int(row, "width"),
            height=_int(row, "height"),
        )
        for row in _object_list(value, "images")
    ]


def _annotations(value: object) -> list[CocoAnnotation]:
    items: list[CocoAnnotation] = []
    for row in _object_list(value, "annotations"):
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Each COCO annotation bbox must contain four values")
        items.append(
            CocoAnnotation(
                image_id=_int(row, "image_id"),
                category_id=_int(row, "category_id"),
                bbox=(
                    _number(bbox[0], "bbox"),
                    _number(bbox[1], "bbox"),
                    _number(bbox[2], "bbox"),
                    _number(bbox[3], "bbox"),
                ),
            )
        )
    return items


def _object_list(value: object, field: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"COCO '{field}' must be a list of objects")
    return value


def _int(row: dict[str, object], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"COCO field '{field}' must be an integer")
    return value


def _str(row: dict[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"COCO field '{field}' must be a non-empty string")
    return value


def _number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"COCO field '{field}' must contain numbers")
    return float(value)


def _normalize_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("COCO image dimensions must be positive")
    x, y, width, height = bbox
    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + width))
    y2 = max(0.0, min(float(image_height), y + height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None
    return (
        (x1 + clipped_width / 2.0) / image_width,
        (y1 + clipped_height / 2.0) / image_height,
        clipped_width / image_width,
        clipped_height / image_height,
    )
