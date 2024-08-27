from typing import Optional

from pydantic import BaseModel


class FeatureMetadata(BaseModel):
    """Feature metadata.

    Attributes:
        question (str): Question text.
    """

    question: Optional[str] = None
