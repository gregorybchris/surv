import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from surv.models.feature import Feature
from surv.models.feature_info import FeatureInfo


@dataclass
class Dataset:
    """Survey dataset."""

    tabular: pd.DataFrame
    feature_info: FeatureInfo

    @classmethod
    def from_files(cls, tabular_filepath: Path, feature_info_filepath: Path) -> "Dataset":
        """Load the dataset from the data directory.

        Args:
            tabular_filepath (Path): Path tabular dataset CSV file.
            feature_info_filepath (Path): Path to feature_info JSON file.
        """
        tabular = pd.read_csv(tabular_filepath)

        with feature_info_filepath.open("r") as file:
            feature_info_json = json.load(file)
            feature_info = FeatureInfo(**feature_info_json)

        return cls(tabular, feature_info)

    def get_column(self, feature_name: str) -> np.ndarray:
        """Get a column from the dataset.

        Args:
            feature_name (str): Name of the feature.

        Returns:
            np.ndarray: Feature data.
        """
        return self.tabular[feature_name].values

    def get_feature(self, feature_name: str) -> Feature:
        """Get a feature from the dataset.

        Args:
            feature_name (str): Name of the feature.

        Returns:
            Feature: The feature for the given name.
        """
        for feature in self.feature_info.features:
            if feature.name == feature_name:
                return feature
        msg = f"Feature '{feature_name}' not found in dataset metadata."
        raise ValueError(msg)

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return self.tabular.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features in the dataset."""
        return self.tabular.shape[1]

    @property
    def n_training_features(self) -> int:
        """Number of training features in the dataset."""
        n = 0
        for feature in self.feature_info.features:
            if feature.name == self.feature_info.target_feature_name:
                continue
            if feature.attributes.identifier:
                continue
            n += 1
        return n
