# X-Ray Inspection Workbench

A v0.1 product skeleton for turning an object detector into a resilient,
human-in-the-loop X-ray inspection workflow. The domain and API depend on a
small `Detector` boundary rather than Ultralytics types, so model runtimes can
be replaced without changing the product contract.

> Current state: the API, domain model, configurable risk policy, SQLite
> persistence skeleton, Streamlit shell, tests, Docker setup, COCO-to-YOLO
> converter, Ultralytics adapter, and CI are scaffolded. No dataset or model
> weights are bundled.

## Architecture

```text
X-ray image -> preprocessing -> Detector -> domain mapping -> risk policy
                                                        -> SQLite -> API/UI
```

The model confidence cutoff and operational risk thresholds are intentionally
separate. The future Ultralytics adapter may emit detections from a low model
threshold (default `0.10`), while `config/risk-thresholds.yaml` decides whether
an item is low risk, requires review, or is high risk.

## Local development

Python 3.12 is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
uvicorn xray_workbench.api.main:app --reload
```

Then open `http://localhost:8000/docs` or check:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

In a second terminal, start the placeholder operator console:

```powershell
streamlit run ui/operator_console/app.py
```

## Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Operator console: `http://localhost:8501`

## Quality checks

```powershell
pytest
ruff check .
mypy src
```

## API (initial contract)

- `GET /api/v1/health`
- `POST /api/v1/inspections` (multipart image; fail-safe without local weights)
- `GET /api/v1/inspections?limit=50`
- `GET /api/v1/inspections/{inspection_id}`
- `GET /api/v1/inspections/{inspection_id}/image`
- `GET /api/v1/inspections/{inspection_id}/events`
- `POST /api/v1/inspections/{inspection_id}/review`

Without `XRAY_MODEL_PATH`, uploads fail safe to `manual_review_required`. A
local weights path activates `YoloDetector`; the adapter never resolves a model
name and therefore does not initiate a weight download.

The Streamlit console is wired to these endpoints: it uploads an image, renders
normalized bounding boxes when detections exist, records one of the three
operator decisions, displays recent inspection history, and reopens the stored
image and audit trail for a previous inspection.

## Prepare a local dataset split

PIDray is not downloaded automatically because its official repository does
not currently include an explicit license file. After obtaining the files and
confirming terms for your use, convert one COCO-format split with:

```powershell
python training/prepare_dataset.py `
  --annotations C:\path\to\annotations.json `
  --images C:\path\to\images `
  --output data\processed\pidray `
  --split train `
  --classes gun knife scissors lighter `
  --max-images 100
```

Repeat for a validation split. Use `--include-backgrounds` deliberately when
negative examples are required; the default keeps only images containing one
of the selected classes.

Once a local base-weight file has been obtained deliberately, run the bounded
smoke pipeline without any implicit download:

```powershell
python training/train.py --model C:\path\to\base-model.pt --epochs 3
python training/evaluate.py --model training\experiments\pidray-smoke\weights\best.pt
```

## Repository map

- `src/xray_workbench/domain`: framework-independent inspection rules.
- `src/xray_workbench/vision`: detector boundary and adapters.
- `src/xray_workbench/application`: orchestration and risk policy.
- `src/xray_workbench/infrastructure`: SQLAlchemy/SQLite persistence.
- `src/xray_workbench/api`: FastAPI routes and wire schemas.
- `training`: dataset, training, and evaluation entry points.
- `ui/operator_console`: Streamlit operator shell.
- `docs`: architecture, domain notes, and ADRs.

## Next milestone

Prepare a small, permitted PIDray subset (3-4 classes), run a 1-3 epoch smoke
training, and prove the first real image upload -> detection -> risk ->
persistence -> review flow. Dataset files and model weights stay outside Git.

## Safety and limitations

This is a learning/portfolio prototype, not certified security equipment. Its
outputs must not be used as the sole basis for real-world safety decisions.
