import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from sqlalchemy import String, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from xray_workbench.infrastructure.database import Base

_CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


@dataclass(frozen=True, slots=True)
class StoredImage:
    inspection_id: UUID
    storage_key: str
    original_filename: str
    content_type: str


class ImageStore(Protocol):
    def save(self, inspection_id: UUID, payload: bytes, content_type: str) -> str: ...

    def read(self, storage_key: str) -> bytes: ...


class ImageReferenceRepository(Protocol):
    def save(self, image: StoredImage) -> None: ...

    def get(self, inspection_id: UUID) -> StoredImage | None: ...


class InspectionImageRecord(Base):
    __tablename__ = "inspection_images"

    inspection_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    storage_key: Mapped[str] = mapped_column(String(255), unique=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))


class LocalImageStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, inspection_id: UUID, payload: bytes, content_type: str) -> str:
        if not payload:
            raise ValueError("Image payload cannot be empty")
        extension = _CONTENT_TYPE_EXTENSIONS.get(content_type)
        if extension is None:
            raise ValueError(f"Unsupported image content type: {content_type}")
        storage_key = f"{inspection_id}{extension}"
        target = self._resolve_key(storage_key)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, target)
        return storage_key

    def read(self, storage_key: str) -> bytes:
        path = self._resolve_key(storage_key)
        if not path.is_file():
            raise FileNotFoundError(storage_key)
        return path.read_bytes()

    def _resolve_key(self, storage_key: str) -> Path:
        if Path(storage_key).name != storage_key:
            raise ValueError("Invalid image storage key")
        path = (self._root / storage_key).resolve()
        if path.parent != self._root:
            raise ValueError("Image storage key escapes the configured root")
        return path


class SqlAlchemyImageReferenceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, image: StoredImage) -> None:
        self._session.merge(
            InspectionImageRecord(
                inspection_id=str(image.inspection_id),
                storage_key=image.storage_key,
                original_filename=image.original_filename,
                content_type=image.content_type,
            )
        )
        self._session.commit()

    def get(self, inspection_id: UUID) -> StoredImage | None:
        record = self._session.scalar(
            select(InspectionImageRecord).where(
                InspectionImageRecord.inspection_id == str(inspection_id)
            )
        )
        if record is None:
            return None
        return StoredImage(
            inspection_id=UUID(record.inspection_id),
            storage_key=record.storage_key,
            original_filename=record.original_filename,
            content_type=record.content_type,
        )
