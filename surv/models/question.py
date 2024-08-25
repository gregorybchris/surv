from pydantic import BaseModel

from surv.models.response_type import ResponseType


class Question(BaseModel):
    """Survey question model.

    Attributes:
        slug (str): Question slub.
        response_type (ResponseType): Response type.
        text (str): Question text.
    """

    slug: str
    response_type: ResponseType
    text: str
