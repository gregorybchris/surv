from pydantic import BaseModel


class FeatureAttributes(BaseModel):
    """Feature attributes.

    Attributes:
        identifier (bool): Whether the feature is a sample identifier.
    """

    identifier: bool = False
