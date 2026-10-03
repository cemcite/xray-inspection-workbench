from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from xray_workbench.api.health_routes import router as health_router
from xray_workbench.api.inspection_routes import router as inspection_router
from xray_workbench.infrastructure.database import create_schema
from xray_workbench.infrastructure.logging import configure_logging
from xray_workbench.settings import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    create_schema()
    yield


settings = get_settings()
app = FastAPI(
    title=settings.application_name,
    version="0.1.0",
    description="Human-in-the-loop X-ray inspection prototype",
    lifespan=lifespan,
)
app.include_router(health_router, prefix="/api/v1")
app.include_router(inspection_router, prefix="/api/v1")


def run() -> None:
    uvicorn.run("xray_workbench.api.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
