from pydantic import BaseModel

from surv.dataset.feature import Feature


class Result(BaseModel):
    """Evaluation result."""


class Continue(Result):
    """Continue evaluation result."""

    feature: Feature
    information_gain: float


class Terminal(Result):
    """Terminal evaluation result."""

    category: str


class Unknown(Result):
    """Unknown evaluation result."""
