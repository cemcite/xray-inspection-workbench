# X-Ray Inspection Workbench

A v0.1 product skeleton for turning an object detector into a resilient,
human-in-the-loop X-ray inspection workflow. The domain and API depend on a
small `Detector` boundary rather than Ultralytics types, so model runtimes can
be replaced without changing the product contract.

> Current state: the API, domain model, configurable risk policy, SQLite
> persistence, Streamlit operator flow, COCO-to-YOLO converter, Ultralytics
> adapter, CI, and a CPU-trained YOLO26n baseline are implemented. Dataset files
> and model weights are local artifacts and are not committed to this repository.

## Architecture

```text
X-ray image -> preprocessing -> Detector -> domain mapping -> risk policy
                                                        -> SQLite -> API/UI
```

The model confidence cutoff and operational risk thresholds are intentionally
separate. The Ultralytics adapter emits detections from a low model threshold
(default `0.10`), while `config/risk-thresholds.yaml` decides whether an item is
low risk, requires review, or is high risk. Image enhancement algorithms remain
an experiment; the current detector receives the decoded image without CLAHE or
denoising.

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
  --balanced `
  --max-images 100
```

When disk space is limited, images can be read directly from an archive without
extracting the full dataset:

```powershell
python training/prepare_dataset.py `
  --annotations C:\Datasets\PIDray\annotations\train.json `
  --image-archive C:\Datasets\PIDray\pidray.zip `
  --archive-prefix pidray/train `
  --output data\processed\pidray `
  --split train `
  --classes gun knife scissors lighter `
  --balanced `
  --max-images 100
```

Repeat for a validation split. Use `--include-backgrounds` deliberately when
negative examples are required; the default keeps only images containing one
of the selected classes.

Once a local base-weight file has been obtained deliberately, run the bounded
smoke pipeline without any implicit download:

```powershell
python training/train.py `
  --model models\base\yolo26n.pt `
  --epochs 1 `
  --image-size 320 `
  --batch-size 4 `
  --workers 0 `
  --device cpu `
  --name pidray-smoke-cpu
python training/evaluate.py --model training\experiments\pidray-smoke-cpu\weights\best.pt
```

The first local smoke run used 100 balanced training images and 40 balanced
validation images. It completed the full train/validation pipeline, but its
one-epoch metrics are intentionally not suitable for operational use. See
`models/manifests/pidray-smoke-cpu.json` for the recorded run details.

A second CPU baseline used 500 training images and 160 validation images, with
20% background examples in each split. Its 20-epoch result reached mAP50
0.4296 and mAP50-95 0.2794. The class-level results and limitations are recorded
in `models/manifests/pidray-baseline-cpu.json`. Follow-up 160-image subset
evaluations reached mAP50 0.284 on hard and 0.0569 on hidden, showing weak
generalization beyond the easy validation subset. These are sampled-subset
results, not metrics for the full PIDray splits. To try it locally, set:

```powershell
$env:XRAY_MODEL_PATH="training\experiments\pidray-baseline-cpu\weights\best.pt"
$env:XRAY_MODEL_VERSION="baseline-20e"
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

The [expanded baseline experiment](docs/expanded-baseline.md) prepares 2,000
training and 400 independent validation images and runs training plus common
checkpoint evaluation. Its dataset audit and live progress are local artifacts;
see the experiment document for paths and commands.

Image-level error diagnostics and a local review gallery are available through
`python training/analyze_errors.py`. See [error analysis](docs/error-analysis.md)
for the measured misses, false alarms, matching policy, and reproduction command.

Improve generalization against hard and hidden PIDray examples through a
documented error analysis. Compare raw input with CLAHE and denoise+CLAHE using
the same evaluation splits and report per-class quality and latency before
choosing any preprocessing default. Dataset files and model weights stay
outside Git.

## Safety and limitations

This is a learning/portfolio prototype, not certified security equipment. Its
outputs must not be used as the sole basis for real-world safety decisions.
