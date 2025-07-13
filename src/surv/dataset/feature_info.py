from pydantic import BaseModel

from surv.dataset.feature import Feature


class FeatureInfo(BaseModel):
    """Dataset feature information.

    Attributes:
        features (list[Feature]): List of dataset features.
        target_feature_name (str): Name of the target feature.
    """

    features: list[Feature]
    target_feature_name: str
