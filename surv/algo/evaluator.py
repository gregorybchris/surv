import logging
from dataclasses import dataclass

import numpy as np

from surv.algo.constraints import Constraint, EqConstraint, GtConstraint, LtConstraint
from surv.algo.result import Continue, Result, Terminal, Unknown
from surv.dataset.dataset import Dataset
from surv.dataset.feature import Feature
from surv.dataset.feature_purpose import Training
from surv.dataset.feature_types import Categorical, FeatureType

logger = logging.getLogger(__name__)


@dataclass
class Evaluator:
    """Feature evaluator."""

    def evaluate(self, dataset: Dataset, constraints: list[Constraint]) -> Result:
        """Evaluate a dataset for the optimal feature data to collect.

        Args:
            dataset (Dataset): Dataset to evaluate.
            constraints (list[Constraint]): Known constraints.
        """
        logger.info("-----")
        logger.info("Evaluating dataset with %i constraints.", len(constraints))
        feature_info = dataset.feature_info

        information_gain_map: dict[str, float] = {}
        for feature in feature_info.features:
            if not isinstance(feature.purpose, Training):
                logger.debug("Skipping non-training feature %s with purpose %s", feature.name, feature.purpose.name)
                continue
            logger.debug("Computing information gain for feature: %s", feature.name)
            information_gain = self._compute_information_gain(dataset, feature, constraints)
            logger.info("Information gain - %s: %f", feature.name, information_gain)
            information_gain_map[feature.name] = information_gain

        max_information_gain = 0.0
        best_feature_name = None
        for feature_name, information_gain in information_gain_map.items():
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_feature_name = feature_name

        if best_feature_name is None:
            mask_column = self._filter_dataset(dataset, constraints)
            target_column = dataset.get_column(dataset.feature_info.target_feature_name)
            target_column = target_column[mask_column]
            unique_targets = np.unique(target_column)
            if unique_targets.shape[0] == 1:
                category = unique_targets[0]
                logger.info("Terminal node reached with category: %s", category)
                return Terminal(category=category)
            logger.info(f"Unknown node reached with targets: {unique_targets}")
            return Unknown()

        best_feature = dataset.get_feature(best_feature_name)
        return Continue(feature=best_feature, information_gain=max_information_gain)

    def _compute_information_gain(self, dataset: Dataset, feature: Feature, constraints: list[Constraint]) -> float:
        match feature.type:
            case Categorical():
                return self._compute_information_gain_categorical(dataset, feature, constraints)
            case _:
                raise NotImplementedError

    @classmethod
    def _filter_dataset(cls, dataset: Dataset, constraints: list[Constraint]) -> np.ndarray:
        """Filter the dataset based on constraints."""
        logger.debug("Initial dataset has %d samples.", dataset.n_samples)
        mask_column = np.ones(dataset.n_samples, dtype=bool)
        for constraint in constraints:
            logger.debug("Applying constraint: %s %s", type(constraint).__name__, constraint)
            constraint_column = dataset.get_column(constraint.feature.name)
            match constraint:
                case EqConstraint(value=value):
                    mask_column &= constraint_column == value
                case LtConstraint(value=value):
                    mask_column &= constraint_column < value
                case GtConstraint(value=value):
                    mask_column &= constraint_column > value
                case _:
                    raise NotImplementedError
            logger.debug("Filtered dataset has %d samples.", np.sum(mask_column))
        return mask_column

    def _compute_information_gain_categorical(
        self,
        dataset: Dataset,
        feature: Feature,
        constraints: list[Constraint],
    ) -> float:
        column = dataset.get_column(feature.name)
        target_feature = dataset.get_feature(dataset.feature_info.target_feature_name)
        target_column = dataset.get_column(dataset.feature_info.target_feature_name)

        mask_column = self._filter_dataset(dataset, constraints)

        # Filter down dataset rows to only those that match the constraints.
        column = column[mask_column]
        target_column = target_column[mask_column]

        entropy_initial = self._compute_entropy(target_column, target_feature.type)
        information_gain = entropy_initial
        for category in np.unique(column):
            target_subset = target_column[column == category]
            subset_entropy = self._compute_entropy(target_subset, target_feature.type)
            proportion = target_subset.shape[0] / target_column.shape[0]
            information_gain -= proportion * subset_entropy
        return information_gain

    def _compute_entropy(self, column: np.ndarray, feature_type: FeatureType) -> float:
        match feature_type:
            case Categorical():
                return self._compute_entropy_categorical(column)
            case _:
                raise NotImplementedError

    @staticmethod
    def _compute_entropy_categorical(column: np.ndarray) -> float:
        h = 0.0
        for x in np.unique(column):
            p_x = np.sum(column == x) / column.shape[0]
            h -= p_x * np.log2(p_x)
        return float(h)
