from pydantic import BaseModel, Field

from surv.models.feature_types import FeatureType


class Feature(BaseModel):
    """Dataset feature model.

    Attributes:
        name (str): Feature name.
        feature_type (FeatureType): Data type of response.
    """

    name: str
    type: FeatureType = Field(..., discriminator="name")

    def __repr__(self) -> str:
        """Return the string representation of the feature."""
        return f"Feature({self.name})"
