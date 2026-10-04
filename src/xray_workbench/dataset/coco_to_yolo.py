import json
import shutil
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile


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
    images_dir: str | Path | None,
    output_dir: str | Path,
    *,
    split: str,
    class_names: Sequence[str],
    include_backgrounds: bool = False,
    max_images: int | None = None,
    overwrite: bool = False,
    image_archive: str | Path | None = None,
    archive_prefix: str = "",
    balanced: bool = False,
    background_fraction: float = 0.0,
) -> ConversionSummary:
    """Convert one COCO detection split to YOLO labels and copied images."""

    if not split or Path(split).name != split:
        raise ValueError("split must be a single non-empty path segment")
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be non-empty and unique")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be greater than zero")
    if not 0.0 <= background_fraction < 1.0:
        raise ValueError("background_fraction must be in [0, 1)")
    if background_fraction and not include_backgrounds:
        raise ValueError("background_fraction requires include_backgrounds=True")
    if background_fraction and max_images is None:
        raise ValueError("background_fraction requires max_images")
    if (images_dir is None) == (image_archive is None):
        raise ValueError("Provide exactly one of images_dir or image_archive")

    annotation_path = Path(annotation_file)
    source_images = Path(images_dir) if images_dir is not None else None
    archive_path = Path(image_archive) if image_archive is not None else None
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
    if balanced:
        selected_images = _balanced_order(
            selected_images,
            annotations_by_image,
            category_to_class,
        )
    if background_fraction:
        selected_images = _reserve_backgrounds(
            selected_images,
            annotations_by_image,
            max_images=max_images,
            background_fraction=background_fraction,
        )
    if max_images is not None:
        selected_images = selected_images[:max_images]

    image_output = destination / "images" / split
    label_output = destination / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    annotation_count = 0
    skipped_count = 0
    archive = ZipFile(archive_path) if archive_path is not None else None
    try:
        for image in selected_images:
            target = image_output / Path(image.file_name).name
            if target.exists() and not overwrite:
                raise FileExistsError(f"Destination already exists: {target}")
            _copy_image(
                image,
                target,
                source_images=source_images,
                archive=archive,
                archive_prefix=archive_prefix,
            )

            lines: list[str] = []
            for annotation in annotations_by_image[image.id]:
                normalized = _normalize_bbox(annotation.bbox, image.width, image.height)
                if normalized is None:
                    skipped_count += 1
                    continue
                class_id = category_to_class[annotation.category_id]
                lines.append(
                    f"{class_id} " + " ".join(f"{value:.8f}" for value in normalized)
                )
                annotation_count += 1
            (label_output / f"{target.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
    finally:
        if archive is not None:
            archive.close()

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


def _balanced_order(
    images: list[CocoImage],
    annotations_by_image: dict[int, list[CocoAnnotation]],
    category_to_class: dict[int, int],
) -> list[CocoImage]:
    buckets: dict[int, deque[CocoImage]] = {
        class_id: deque() for class_id in sorted(set(category_to_class.values()))
    }
    backgrounds: list[CocoImage] = []
    for image in images:
        class_ids = {
            category_to_class[annotation.category_id]
            for annotation in annotations_by_image[image.id]
            if annotation.category_id in category_to_class
        }
        if not class_ids:
            backgrounds.append(image)
            continue
        for class_id in sorted(class_ids):
            buckets[class_id].append(image)

    ordered: list[CocoImage] = []
    seen: set[int] = set()
    while any(buckets.values()):
        for class_id in sorted(buckets):
            bucket = buckets[class_id]
            while bucket and bucket[0].id in seen:
                bucket.popleft()
            if bucket:
                image = bucket.popleft()
                ordered.append(image)
                seen.add(image.id)
    ordered.extend(image for image in backgrounds if image.id not in seen)
    return ordered


def _reserve_backgrounds(
    images: list[CocoImage],
    annotations_by_image: dict[int, list[CocoAnnotation]],
    *,
    max_images: int | None,
    background_fraction: float,
) -> list[CocoImage]:
    if max_images is None:
        raise ValueError("max_images is required when reserving backgrounds")
    foregrounds = [image for image in images if annotations_by_image[image.id]]
    backgrounds = [image for image in images if not annotations_by_image[image.id]]
    background_count = min(round(max_images * background_fraction), len(backgrounds))
    foreground_count = min(max_images - background_count, len(foregrounds))
    selected = foregrounds[:foreground_count] + backgrounds[:background_count]
    return sorted(selected, key=lambda image: image.file_name)


def _copy_image(
    image: CocoImage,
    target: Path,
    *,
    source_images: Path | None,
    archive: ZipFile | None,
    archive_prefix: str,
) -> None:
    if source_images is not None:
        source = source_images / image.file_name
        if not source.is_file():
            raise FileNotFoundError(f"COCO image is missing: {source}")
        shutil.copy2(source, target)
        return
    if archive is None:
        raise RuntimeError("No image source is configured")
    member = str(PurePosixPath(archive_prefix) / PurePosixPath(image.file_name))
    try:
        payload = archive.read(member)
    except KeyError as error:
        raise FileNotFoundError(f"COCO image is missing from archive: {member}") from error
    target.write_bytes(payload)


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
