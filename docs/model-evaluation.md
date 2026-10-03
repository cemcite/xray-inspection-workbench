# Model evaluation

The first baseline must report per-class precision, recall, F1, mAP@50,
mAP@50:95, and preprocessing/inference/postprocessing latency. Preserve a
versioned model manifest alongside every result.

Preprocessing changes are experiments, not defaults. Compare at least raw,
CLAHE, and denoise+CLAHE pipelines before adopting one. Document false
positives, false negatives, low-confidence detections, and hidden objects.
