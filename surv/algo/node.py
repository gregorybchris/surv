from dataclasses import dataclass
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from surv.models.question import Question


class EqCondition(BaseModel):
    """Equals condition."""

    type: Literal["eq"]
    value: str


class LtCondition(BaseModel):
    """Less than condition."""

    type: Literal["lt"]
    value: float


class GtCondition(BaseModel):
    """Greater than condition."""

    type: Literal["gt"]
    value: float


class Condition(BaseModel):
    """Condition to split the data on."""

    comparison: Union[EqCondition, LtCondition, GtCondition] = Field(..., discriminator="type")


@dataclass
class Node:
    """Node in the decision tree."""

    question: Question
    condition: Condition
    left: Optional["Node"] = None
    right: Optional["Node"] = None
