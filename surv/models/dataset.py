import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from surv.models.feature import Feature
from surv.models.feature_info import FeatureInfo
from surv.models.feature_purpose import Training
from surv.models.feature_types import Categorical


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

        tabular = cls.apply_aliases(feature_info, tabular)

        return cls(tabular, feature_info)

    @classmethod
    def apply_aliases(cls, feature_info: FeatureInfo, tabular: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical numeric aliases to the tabular dataset.

        Args:
            feature_info (FeatureInfo): Feature metadata.
            tabular (pd.DataFrame): Tabular dataset.

        Returns:
            pd.DataFrame: Tabular dataset with aliased columns.
        """
        tabular = tabular.copy()
        for feature in feature_info.features:
            if isinstance(feature.type, Categorical) and feature.type.numeric_alias:
                column = tabular[feature.name]
                categories = feature.type.categories
                for x in column:
                    if x < 1 or x > len(categories):
                        msg = f"Value '{x}' not in categories for feature '{feature.name}'."
                        raise ValueError(msg)
                tabular[feature.name] = column.apply(lambda x, categories=categories: categories[x - 1])
        return tabular

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
            if isinstance(feature.purpose, Training):
                n += 1
        return n

    def validate(self) -> None:
        """Validate the dataset."""
        if set(self.tabular.columns) != {feature.name for feature in self.feature_info.features}:
            msg = "Feature names in the dataset do not match feature metadata."
            raise ValueError(msg)

        for feature in self.feature_info.features:
            if isinstance(feature.type, Categorical):
                column = self.get_column(feature.name)
                for value in column:
                    if value not in feature.type.categories:
                        msg = f"Value '{value}' not in categories for feature '{feature.name}'."
                        raise ValueError(msg)
