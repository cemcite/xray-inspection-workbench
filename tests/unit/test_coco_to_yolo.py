import json
from pathlib import Path

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
