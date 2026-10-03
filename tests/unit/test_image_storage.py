from pathlib import Path
from uuid import uuid4

import pytest

from xray_workbench.infrastructure.image_storage import LocalImageStore


def test_local_image_store_round_trip(tmp_path: Path) -> None:
    store = LocalImageStore(tmp_path)
    inspection_id = uuid4()

    key = store.save(inspection_id, b"png-payload", "image/png")

    assert key == f"{inspection_id}.png"
    assert store.read(key) == b"png-payload"


def test_local_image_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = LocalImageStore(tmp_path)

    with pytest.raises(ValueError, match="storage key"):
        store.read("../outside.png")


def test_local_image_store_rejects_unknown_content_type(tmp_path: Path) -> None:
    store = LocalImageStore(tmp_path)

    with pytest.raises(ValueError, match="Unsupported"):
        store.save(uuid4(), b"payload", "image/gif")
