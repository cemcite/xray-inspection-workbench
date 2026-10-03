# ADR-001: Isolate the model runtime

Status: accepted

The application depends on a local `Detector` protocol. Ultralytics-specific
objects remain inside `YoloDetector`. This makes tests deterministic and leaves
an explicit migration path to ONNX Runtime or another detector.
