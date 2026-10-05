# Expanded PIDray baseline experiment

## Dataset prepared on 2026-10-05

`training/prepare_expanded_baseline.py` reads the local PIDray ZIP and
`train.json`, intersects available filenames, and converts only those images.
The annotation file lists 76,913 training images; this local ZIP provides
29,457 of them. Source image dimensions were checked against annotations for
all selected files. No new dataset or model downloads were required.

| Split | Images | Background images | Gun instances | Knife instances | Scissors instances | Lighter instances |
|---|---:|---:|---:|---:|---:|---:|
| Train | 2,000 | 400 | 426 | 419 | 460 | 405 |
| Validation | 400 | 80 | 94 | 88 | 84 | 80 |

The new train subset retains all 500 original baseline training images and
adds 1,500 images using the same deterministic class-balanced ordering.
Validation excludes selected training filenames. SHA256 checks found zero
image-content overlap for train/validation, train/hard-hidden, and
validation/hard-hidden. Train and validation have 2,000 and 400 distinct
image-content hashes respectively. "Background" means no labels for the four
selected classes; other PIDray classes may be present.

Local data is under `data/processed/pidray-expanded`. Split manifests record
selected filenames, annotation source/hash, annotation counts and background
counts. `manifests/audit.json` records the subset checks and composition.

## Training and comparison

Training started at 2026-10-05 05:20:29 UTC. Its results are pending; preparation
and startup do not establish improved model performance.

The first epoch completed in 285.8 seconds including setup and validation;
`best.pt` and `last.pt` were created, and epoch 2 started. The initial remaining
training estimate was 5,430 seconds (about 90 minutes). This is an early estimate;
read `state.json` for current progress. The process continues in the background.

The run starts from the same local `models/base/yolo26n.pt` pretrained weights,
not the original baseline checkpoint. Parameters: 20 epochs, 320px, batch 16,
CPU, workers 0, seed 42, deterministic mode, and no CLAHE/denoising. Standard
Ultralytics training augmentation remains enabled, as in the original baseline.

The validation source changes from the original easy subset to 400 held-out
`train.json` images. This changes checkpoint selection as well as training
volume; the experiment cannot attribute every difference to volume alone.
Both best checkpoints are therefore re-evaluated on the same new validation
set and the unchanged 160-image hard/hidden subsets. A strict volume-only
ablation would also retrain the 500-image reference using this validation set.

`training/run_expanded_baseline.py` runs the following sequence automatically:

1. Check the prepared dataset audit and local weight files.
2. Train 20 epochs and record progress after each epoch.
3. Validate original and expanded best checkpoints on the three common splits.
4. Generate fixed-threshold error reports/galleries for both checkpoints on
   hard/hidden (confidence 0.10, IoU 0.50).
5. Write `comparison.json` with quality deltas, hashes and dataset metadata.

The active run directory is `runs/expanded-baseline-20261005T052029Z`.
`state.json` records the phase, worker PID, completed epochs, estimated
remaining training time, checkpoint paths and any failure traceback.
`stdout.log` and `stderr.log` contain process output. The model experiment is
`training/experiments/pidray-expanded-cpu`; the run preserves the original
checkpoint and all earlier experiments.

Reproduction commands (use fresh output/run directories; existing training
directories are refused):

```powershell
.venv\Scripts\python.exe training/prepare_expanded_baseline.py
.venv\Scripts\python.exe training/run_expanded_baseline.py `
  --run-dir runs/expanded-baseline-reproduction
```

Inspect active progress:

```powershell
Get-Content runs/expanded-baseline-20261005T052029Z/state.json
Get-Content runs/expanded-baseline-20261005T052029Z/stdout.log -Tail 8
```

Hard/hidden remain diagnostic subsets used in iterative development, so a
separate untouched holdout is still needed for a final generalization claim.
Dataset images, checkpoints and runtime outputs stay in ignored local folders.
