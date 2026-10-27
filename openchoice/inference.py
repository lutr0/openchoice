from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Decision:
    instructions: str
    options: tuple[str, ...]
    primitive: Literal["choice", "score", "noul"]
    true_criterion: str | None = None
    false_criterion: str | None = None

@dataclass(frozen=True)
class BatchPrediction:
    model: str
    logits: list[list[float]]
    input_tokens: int

class InputTooLong(ValueError):
    """Input exceeds the model's explicit information budget."""

class UnsupportedDecision(ValueError):
    """Request exceeds native decision model capabilities."""

class ResourceExhausted(RuntimeError):
    """Inference could not obtain device memory; the same request may be retried later."""

class Cancelled(RuntimeError):
    """The client stopped waiting, so the server abandoned the remaining work."""
