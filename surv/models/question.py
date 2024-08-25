from pydantic import BaseModel

from surv.models.feature import Feature


class Question(BaseModel):
    """Survey question model.

    Attributes:
        text (str): Question text.
        feature (Feature): Question feature.
    """

    text: str
    feature: Feature

    def __repr__(self) -> str:
        """Return the string representation of the question."""
        return f"Question({self.feature.name})"
