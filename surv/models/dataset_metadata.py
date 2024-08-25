from pydantic import BaseModel

from surv.models.question import Question


class DatasetMetadata(BaseModel):
    """Dataset metadata model.

    Attributes:
        questions (list[Question]): List of survey questions.
        target_feature_name (str): Name of the target feature.
    """

    questions: list[Question]
    target_feature_name: str
