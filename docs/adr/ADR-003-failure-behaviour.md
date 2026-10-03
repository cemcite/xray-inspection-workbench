# ADR-003: Fail safe when the detector is unavailable

Status: accepted

A model-runtime failure moves a pending inspection to
`manual_review_required`. The service remains available and exposes degraded
health. Production use will require explicit alerting, authorization, and audit
requirements before this policy can be considered complete.
