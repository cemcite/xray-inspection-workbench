# ADR-002: Human review is a domain workflow

Status: accepted

Detections do not finalize a threat decision. An operator can confirm a threat,
mark a false positive, or request further review. The inspection entity owns
and validates these transitions so API and UI behavior remain consistent.
