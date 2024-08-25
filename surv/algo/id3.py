from dataclasses import dataclass
from typing import Optional

import numpy as np
from rich.pretty import pprint

from surv.algo.node import Node
from surv.models.dataset import Dataset
from surv.models.feature import Feature
from surv.models.feature_types import Categorical, Datetime, Numeric, Text


@dataclass
class Id3:
    """ID3 Decision Tree Algorithm."""

    max_depth: int = 3
    min_leaf_size: int = 1
    _node: Optional[Node] = None

    def fit(self, dataset: Dataset) -> None:
        """Fit the model to the data.

        Args:
            dataset (Dataset): Dataset to fit the model to.
        """
        metadata = dataset.metadata
        questions = metadata.questions
        target_feature_name = dataset.metadata.target_feature_name

        information_gain_map: dict[str, float] = {}
        for question in questions:
            question_feature = question.feature
            if question_feature.name == target_feature_name:
                continue
            information_gain = self._compute_information_gain(dataset, question_feature)
            information_gain_map[question_feature.name] = information_gain

        pprint(information_gain_map)

    def _compute_entropy(self, feature: Feature, column: np.ndarray) -> float:
        match feature.feature_type.metadata:
            case Categorical():
                return self._compute_entropy_categorical(column)
            case Numeric():
                raise NotImplementedError
            case Datetime():
                raise NotImplementedError
            case Text():
                raise NotImplementedError

    def _compute_entropy_categorical(self, column: np.ndarray) -> float:  # noqa: PLR6301
        entropy = 0.0
        for target in np.unique(column):
            proportion = np.sum(column == target) / column.shape[0]
            entropy -= proportion * np.log2(proportion)
        return float(entropy)

    def _compute_information_gain(self, dataset: Dataset, feature: Feature) -> float:
        match feature.feature_type.metadata:
            case Categorical():
                return self._compute_information_gain_categorical(dataset, feature)
            case Numeric():
                raise NotImplementedError
            case Datetime():
                raise NotImplementedError
            case Text():
                raise NotImplementedError

    def _compute_information_gain_categorical(self, dataset: Dataset, feature: Feature) -> float:
        target_feature = dataset.get_feature(dataset.metadata.target_feature_name)
        target_column = dataset.get_column(dataset.metadata.target_feature_name)
        entropy_initial = self._compute_entropy(target_feature, target_column)

        information_gain = entropy_initial
        question_column = dataset.get_column(feature.name)
        for category in np.unique(question_column):
            target_subset = target_column[question_column == category]
            subset_entropy = self._compute_entropy(target_feature, target_subset)
            proportion = target_subset.shape[0] / target_column.shape[0]
            information_gain -= proportion * subset_entropy

        return information_gain

    def predict(self, dataset: Dataset) -> Dataset:
        """Predict the target variable.

        Returns:
            Dataset: Dataset with the predicted target variable.
        """
        raise NotImplementedError
