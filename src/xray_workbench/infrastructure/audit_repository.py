from datetime import datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import JSON, DateTime, String, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from xray_workbench.domain.audit import AuditEvent, AuditEventType
from xray_workbench.infrastructure.database import Base


class AuditRepository(Protocol):
    def append(self, event: AuditEvent) -> None: ...

    def list_for_inspection(self, inspection_id: UUID) -> list[AuditEvent]: ...


class AuditEventRecord(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    inspection_id: Mapped[str] = mapped_column(String(36), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    message: Mapped[str] = mapped_column(String(500))
    details: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class SqlAlchemyAuditRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def append(self, event: AuditEvent) -> None:
        self._session.add(
            AuditEventRecord(
                id=str(event.id),
                inspection_id=str(event.inspection_id),
                event_type=event.event_type.value,
                message=event.message,
                details=dict(event.details),
                occurred_at=event.occurred_at,
            )
        )
        self._session.commit()

    def list_for_inspection(self, inspection_id: UUID) -> list[AuditEvent]:
        records = self._session.scalars(
            select(AuditEventRecord)
            .where(AuditEventRecord.inspection_id == str(inspection_id))
            .order_by(AuditEventRecord.occurred_at.asc())
        ).all()
        return [
            AuditEvent(
                id=UUID(record.id),
                inspection_id=UUID(record.inspection_id),
                event_type=AuditEventType(record.event_type),
                message=record.message,
                details={key: _detail_value(value) for key, value in record.details.items()},
                occurred_at=record.occurred_at,
            )
            for record in records
        ]


def _detail_value(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
