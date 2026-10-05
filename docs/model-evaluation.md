# Model evaluation

The first baseline must report per-class precision, recall, F1, mAP@50,
mAP@50:95, and preprocessing/inference/postprocessing latency. Preserve a
versioned model manifest alongside every result.

Preprocessing changes are experiments, not defaults. Compare at least raw,
CLAHE, and denoise+CLAHE pipelines before adopting one. Document false
positives, false negatives, low-confidence detections, and hidden objects.

Run the fixed-subset comparison (default: local CPU baseline, 320px, hard and
hidden 160-image subsets):

```powershell
python training/compare_preprocessing.py
```

The command writes transformed images, Ultralytics run artifacts, and
`report.json`/`report.csv` under a new `runs/preprocessing-comparison-*`
directory. Each mode uses the same source images and copied labels. The report
separates the extra CLAHE/denoising time (CPU transform only; decode and disk
I/O excluded) from the validation pipeline timings and metrics. Pass
`--model`, `--data-root`, `--output-dir`, `--image-size`, or `--device` to
select local alternatives. Results on these subsets remain diagnostic, not
full-split benchmarks.

## Preprocessing comparison (2026-10-05)

The same CPU YOLO26n baseline was evaluated on the same 160 hard and 160 hidden
images at 320px. CLAHE and denoise+CLAHE images were generated from the source
images without changing labels. Added enhancement time is measured separately
on CPU and excludes decode/write time. The Ultralytics per-image inference
timings in the local report are single-pass diagnostics; use repeated, warmed runs before
making a latency claim.

| Split | Mode | Precision | Recall | mAP50 | mAP50:95 | Enhancement ms/image |
|---|---|---:|---:|---:|---:|---:|
| Hard | Raw | 0.4540 | 0.2567 | 0.2842 | 0.1634 | 0.00 |
| Hard | CLAHE | 0.3876 | 0.2507 | 0.2625 | 0.1530 | 7.81 |
| Hard | Denoise+CLAHE | 0.3321 | 0.2474 | 0.2460 | 0.1460 | 1,098.98 |
| Hidden | Raw | 0.1627 | 0.0778 | 0.0569 | 0.0302 | 0.00 |
| Hidden | CLAHE | 0.0938 | 0.1029 | 0.0728 | 0.0402 | 5.58 |
| Hidden | Denoise+CLAHE | 0.3156 | 0.1094 | 0.0741 | 0.0390 | 1,174.07 |

Neither enhancement improves the hard subset. On hidden, CLAHE improves mAP
but reduces precision; denoise+CLAHE improves both precision and recall but
still leaves recall at 0.1094. Both remain far too weak to support operational
use; denoising also adds about 1.1 seconds per image on the measured CPU. Keep
raw as the inference default. These small subsets are diagnostic only and do
not establish statistical significance or full-PIDray performance. Raw remains
poor on hidden data, so data/model generalization—not preprocessing—is the
main unresolved issue. The complete local report (including per-class mAP and
runtime metadata) is `runs/preprocessing-comparison-20261005T044654Z/report.json`.

Metric correction: the original comparison script labelled Ultralytics
`box.maps` as per-class mAP50, but those values are mAP50:95. The historical
JSON/CSV column labels have been corrected; aggregate results above are
unaffected. New comparisons record AP50 and AP50:95 separately using
`ap_class_index` to map classes, with absent classes marked as unmeasured.

See [fixed-threshold error analysis](error-analysis.md) for image-level
TP/FP/FN, background false alarms, and the local review gallery. This uses
confidence 0.10 and IoU 0.50; its micro-averaged scores are distinct from the
validation scores above.

## PIDray baseline subset (2026-10-05)

The YOLO26n baseline was trained for 20 CPU epochs at 320px on 500 balanced
PIDray images (400 foreground, 100 background). Validation used 160 images
(128 foreground, 32 background). The sampled easy validation subset reached
mAP50 0.4296 and mAP50:95 0.2794.

The same model was evaluated without retraining on independently sampled,
class-balanced 160-image subsets from the hard and hidden annotations. Hard
reached mAP50 0.284 and mAP50:95 0.163. Hidden reached mAP50 0.0569 and
mAP50:95 0.0302. On hidden, knife mAP50 was 0.0032 and lighter mAP50 was
0.0000156, indicating severe generalization failure for those classes.

These subsets include a requested 20% background share and are diagnostic
samples only; they are not full-split benchmark results. See
`models/manifests/pidray-baseline-cpu.json` for per-class metrics and the weight
hash. Treat this model as experimental, not for operational screening.
