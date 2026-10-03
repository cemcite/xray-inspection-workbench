FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && pip install .

COPY config ./config
COPY ui ./ui

EXPOSE 8000 8501
CMD ["uvicorn", "xray_workbench.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
