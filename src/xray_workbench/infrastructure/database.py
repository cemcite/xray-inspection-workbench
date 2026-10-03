from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from xray_workbench.settings import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine: Engine = create_engine(settings.database_url, connect_args=connect_args)
SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)


def create_schema() -> None:
    # Import registers mappings before create_all.
    from xray_workbench.infrastructure import (  # noqa: F401
        audit_repository,
        image_storage,
        repositories,
    )

    Base.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    with SessionFactory() as session:
        yield session
