from pydantic import BaseModel


class FeatureMetadata(BaseModel):
    """Feature metadata.

    Attributes:
        question (str): Question text.
    """

    question: str
