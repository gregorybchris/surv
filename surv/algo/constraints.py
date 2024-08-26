from typing import Literal, Union

from pydantic import BaseModel, Field

from surv.models.feature import Feature


class EqConstraint(BaseModel):
    """Equals constraint."""

    type: Literal["eq"]
    value: str


class LtConstraint(BaseModel):
    """Less than constraint."""

    type: Literal["lt"]
    value: float


class GtConstraint(BaseModel):
    """Greater than constraint."""

    type: Literal["gt"]
    value: float


class Constraint(BaseModel):
    """Constraint to split the data on."""

    feature: Feature
    comparison: Union[EqConstraint, LtConstraint, GtConstraint] = Field(..., discriminator="type")
