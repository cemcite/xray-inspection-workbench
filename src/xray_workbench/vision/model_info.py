from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    version: str
    ready: bool
