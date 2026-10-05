# Model evaluation

The first baseline must report per-class precision, recall, F1, mAP@50,
mAP@50:95, and preprocessing/inference/postprocessing latency. Preserve a
versioned model manifest alongside every result.

Preprocessing changes are experiments, not defaults. Compare at least raw,
CLAHE, and denoise+CLAHE pipelines before adopting one. Document false
positives, false negatives, low-confidence detections, and hidden objects.

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
