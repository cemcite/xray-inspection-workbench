# Domain model

An `Inspection` owns its detections and optional operator review. Status changes
are methods on the entity so callers cannot bypass transition rules.

```text
created -> processing -> cleared
                    \-> requires_review -> cleared
                                        \-> threat_confirmed
                                        \-> requires_review
                    \-> manual_review_required -> same review outcomes
```

`Detection.confidence` is model evidence. `Detection.risk_level` is the result
of configurable operational policy. They are deliberately different concepts.

Bounding boxes use normalized XYWH coordinates in `[0, 1]` and must remain
inside image bounds.
