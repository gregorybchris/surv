from typing import Optional

from pydantic import BaseModel, Field

from surv.dataset.feature_metadata import FeatureMetadata
from surv.dataset.feature_purpose import FeaturePurpose
from surv.dataset.feature_types import FeatureType


class Feature(BaseModel):
    """Dataset feature model.

    Attributes:
        name (str): Feature name.
        feature_type (FeatureType): Data type of response.
    """

    name: str
    type: FeatureType = Field(..., discriminator="name")
    purpose: FeaturePurpose = Field(..., discriminator="name")
    metadata: Optional[FeatureMetadata] = None

    def __repr__(self) -> str:
        """Return the string representation of the feature."""
        return f"Feature({self.name})"
