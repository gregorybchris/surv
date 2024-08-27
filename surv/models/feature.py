from typing import Optional

from pydantic import BaseModel, Field

from surv.models.feature_attributes import FeatureAttributes
from surv.models.feature_metadata import FeatureMetadata
from surv.models.feature_types import FeatureType


class Feature(BaseModel):
    """Dataset feature model.

    Attributes:
        name (str): Feature name.
        feature_type (FeatureType): Data type of response.
    """

    name: str
    type: FeatureType = Field(..., discriminator="name")
    attributes: FeatureAttributes = FeatureAttributes()
    metadata: Optional[FeatureMetadata] = None

    def __repr__(self) -> str:
        """Return the string representation of the feature."""
        return f"Feature({self.name})"
