import logging
from dataclasses import dataclass

import numpy as np

from surv.algo.constraints import Constraint
from surv.models.dataset import Dataset
from surv.models.feature import Feature
from surv.models.feature_types import Categorical, Datetime, FeatureType, Numeric, Text

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Evaluation result."""

    feature: Feature
    information_gain: float


@dataclass
class Evaluator:
    """Feature evaluator."""

    def evaluate(self, dataset: Dataset, constraints: list[Constraint]) -> EvaluationResult:
        """Evaluate a dataset for the optimal feature data to collect.

        Args:
            dataset (Dataset): Dataset to evaluate.
            constraints (list[Constraint]): Known constraints.
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

        max_information_gain = 0.0
        best_feature_name = None
        for feature_name, information_gain in information_gain_map.items():
            logger.info(f"Feature: {feature_name}, Information Gain: {information_gain}")
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_feature_name = feature_name

        if best_feature_name is None:
            msg = "No best feature found."
            raise ValueError(msg)

        best_feature = dataset.get_feature(best_feature_name)
        return EvaluationResult(feature=best_feature, information_gain=max_information_gain)

    def _compute_entropy(self, column: np.ndarray, feature_type: FeatureType) -> float:
        match feature_type:
            case Categorical():
                return self._compute_entropy_categorical(column)
            case Numeric():
                raise NotImplementedError
            case Datetime():
                raise NotImplementedError
            case Text():
                raise NotImplementedError

    def _compute_entropy_categorical(self, column: np.ndarray) -> float:  # noqa: PLR6301
        h = 0.0
        for x in np.unique(column):
            p_x = np.sum(column == x) / column.shape[0]
            h -= p_x * np.log2(p_x)
        return float(h)

    def _compute_information_gain(self, dataset: Dataset, feature: Feature) -> float:
        match feature.type:
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
        entropy_initial = self._compute_entropy(target_column, target_feature.type)

        information_gain = entropy_initial
        question_column = dataset.get_column(feature.name)
        for category in np.unique(question_column):
            target_subset = target_column[question_column == category]
            subset_entropy = self._compute_entropy(target_subset, target_feature.type)
            proportion = target_subset.shape[0] / target_column.shape[0]
            information_gain -= proportion * subset_entropy

        return information_gain
