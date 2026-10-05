# Fixed-threshold PIDray error analysis

## Reproduce

```powershell
.venv\Scripts\python.exe training/analyze_errors.py --device cpu
```

Defaults: the local baseline `best.pt`, raw input, 320px inference, confidence
0.10 and IoU 0.50. Optional `--model`, `--data-root`, `--output-dir`,
`--confidence`, `--iou`, `--image-size`, and `--device` configure the run.
Existing output directories are refused. The command validates labels, checks
the model class order, and records model and input-image/label hashes.

Outputs under a new `runs/error-analysis-*` directory:

- `report.json`: per-image boxes, matched indices, TP/FP/FN, per-class scores,
  thresholds, hashes, and runtime metadata.
- `cases.csv`: one row per image for locating failures.
- `gallery.html` and `previews/`: up to six cases with the most missed objects
  and six with the most false positives per split; duplicates removed.
  Green boxes show labels, orange boxes show predictions and confidence.

Matching is confidence-ordered, class-aware, greedy, and one-to-one. A
prediction must overlap a same-class annotation at IoU >= 0.50. Duplicate
predictions count as false positives. A wrong-class prediction contributes
one FP and one FN. Predictions below the confidence threshold are excluded.
Precision/recall/F1 aggregate counts across the four classes (micro average);
they differ from Ultralytics validation's class-averaged scores and selected
confidence operating point. This diagnostic does not calculate mAP.

## Baseline run, 2026-10-05

The baseline weight SHA256 is
`8c13cc916e54eff58cab9e8d3e6245840c31fda70913a72bbf42e41220c11930`.
Both splits used their existing 160-image subsets. The local result is
`runs/error-analysis-20261005T050528Z/report.json`; the gallery contains 23
distinct examples. Two generated previews were visually checked.

| Split | TP | FP | FN | Precision | Recall | F1 | Background images with FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hard | 77 | 206 | 134 | 0.2721 | 0.3649 | 0.3117 | 24 / 32 |
| Hidden | 13 | 123 | 115 | 0.0956 | 0.1016 | 0.0985 | 21 / 32 |

| Hidden class | TP | FP | FN | Recall |
|---|---:|---:|---:|---:|
| Gun | 11 | 80 | 21 | 0.3438 |
| Knife | 0 | 15 | 32 | 0.0000 |
| Scissors | 2 | 5 | 30 | 0.0625 |
| Lighter | 0 | 23 | 32 | 0.0000 |

In `xray_hidden00001.png`, the labelled gun has no prediction at the selected
threshold. In `xray_hard00014.png`, predictions call objects gun/lighter;
the original PIDray `test_hard.json` annotations identify Baton and HandCuffs.
Those classes are excluded from this model's four-class subset. Consequently,
"background" means no selected-class labels; it does not mean an empty bag or
the absence of suspicious objects. False alarms here are relative to the
four selected annotation classes.

## Next experiment

The measurements establish missed objects and false alarms, but do not isolate
whether resolution, overlap, sampling, or training volume caused them. Expand
and inspect the training subset from PIDray `train.json`, track class and
background composition, then compare checkpoints with these same diagnostics.
Keep hard/hidden evaluation images out of training. These subsets have already
been used for iterative decisions; retain a separate untouched holdout before
making final generalization claims. Test higher input resolution as a separate
controlled experiment if the class-level failures remain.

Runtime in this report includes image decoding and model startup and is not a
controlled latency benchmark. Images, weights, and generated galleries remain
local assets under ignored directories.
