import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from surv.models.dataset_metadata import DatasetMetadata
from surv.models.feature import Feature


@dataclass
class Dataset:
    """Survey dataset."""

    tabular: pd.DataFrame
    metadata: DatasetMetadata

    @classmethod
    def from_files(cls, tabular_filepath: Path, metadata_filepath: Path) -> "Dataset":
        """Load the dataset from the data directory.

        Args:
            tabular_filepath (Path): Path tabular dataset CSV file.
            metadata_filepath (Path): Path to metadata JSON file.
        """
        tabular = pd.read_csv(tabular_filepath)

        with metadata_filepath.open("r") as file:
            metadata_json = json.load(file)
            metadata = DatasetMetadata(**metadata_json)

        return cls(tabular, metadata)

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
        for question in self.metadata.questions:
            if question.feature.name == feature_name:
                return question.feature
        msg = f"Feature '{feature_name}' not found in dataset metadata."
        raise ValueError(msg)

    @property
    def n_features(self) -> int:
        """Number of features in the dataset."""
        return len(self.metadata.questions)
