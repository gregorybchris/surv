from pydantic import BaseModel

from surv.models.feature import Feature


class Constraint(BaseModel):
    """Constraint to split the data on."""

    feature: Feature


class EqConstraint(Constraint):
    """Equals constraint."""

    value: str


class LtConstraint(Constraint):
    """Less than constraint."""

    value: float


class GtConstraint(Constraint):
    """Greater than constraint."""

    value: float
