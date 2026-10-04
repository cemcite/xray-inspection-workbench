import json
from pathlib import Path
from zipfile import ZipFile

from xray_workbench.dataset import convert_coco_split


def test_converts_selected_coco_category_to_yolo(tmp_path: Path) -> None:
    images = tmp_path / "source-images"
    images.mkdir()
    (images / "selected.png").write_bytes(b"image-placeholder")
    (images / "background.png").write_bytes(b"image-placeholder")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "selected.png",
                        "width": 100,
                        "height": 200,
                    },
                    {
                        "id": 2,
                        "file_name": "background.png",
                        "width": 100,
                        "height": 200,
                    },
                ],
                "categories": [
                    {"id": 2, "name": "Gun"},
                    {"id": 4, "name": "Knife"},
                    {"id": 9, "name": "Hammer"},
                ],
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 4, "bbox": [10, 20, 30, 40]},
                    {"id": 11, "image_id": 1, "category_id": 9, "bbox": [1, 2, 3, 4]},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    summary = convert_coco_split(
        annotations,
        images,
        output,
        split="train",
        class_names=["gun", "knife"],
    )

    assert summary.image_count == 1
    assert summary.annotation_count == 1
    assert (output / "images" / "train" / "selected.png").is_file()
    assert not (output / "images" / "train" / "background.png").exists()
    assert (output / "labels" / "train" / "selected.txt").read_text(
        encoding="utf-8"
    ) == "1 0.25000000 0.20000000 0.30000000 0.20000000\n"


def test_can_include_background_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "background.png").write_bytes(b"image-placeholder")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "background.png",
                        "width": 10,
                        "height": 10,
                    }
                ],
                "categories": [{"id": 1, "name": "knife"}],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )

    summary = convert_coco_split(
        annotations,
        images,
        tmp_path / "output",
        split="val",
        class_names=["knife"],
        include_backgrounds=True,
    )

    assert summary.image_count == 1
    assert (tmp_path / "output" / "labels" / "val" / "background.txt").read_text(
        encoding="utf-8"
    ) == ""


def test_reserves_requested_fraction_for_backgrounds(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image_rows = []
    annotation_rows = []
    for image_id in range(1, 11):
        file_name = f"image-{image_id:02d}.png"
        (images / file_name).write_bytes(b"image-placeholder")
        image_rows.append(
            {"id": image_id, "file_name": file_name, "width": 10, "height": 10}
        )
        if image_id <= 8:
            annotation_rows.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [1, 1, 2, 2],
                }
            )
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": image_rows,
                "categories": [{"id": 1, "name": "knife"}],
                "annotations": annotation_rows,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    summary = convert_coco_split(
        annotations,
        images,
        output,
        split="train",
        class_names=["knife"],
        include_backgrounds=True,
        max_images=10,
        background_fraction=0.2,
    )

    empty_labels = [
        path
        for path in (output / "labels" / "train").glob("*.txt")
        if not path.read_text(encoding="utf-8")
    ]
    assert summary.image_count == 10
    assert len(empty_labels) == 2


def test_reads_selected_images_directly_from_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / "images.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("pidray/train/sample.png", b"png-from-archive")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "sample.png", "width": 20, "height": 10}
                ],
                "categories": [{"id": 1, "name": "knife"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 1, 4, 2]}
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = convert_coco_split(
        annotations,
        None,
        tmp_path / "output",
        split="train",
        class_names=["knife"],
        image_archive=archive_path,
        archive_prefix="pidray/train",
    )

    assert summary.image_count == 1
    assert (tmp_path / "output" / "images" / "train" / "sample.png").read_bytes() == (
        b"png-from-archive"
    )
