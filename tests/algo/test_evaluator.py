from typing import TYPE_CHECKING

import pytest
from surv.algo.evaluator import Evaluator

from tests.algo.conftest import DatasetTag, EvaluationDataset

if TYPE_CHECKING:
    from surv.algo.constraints import Constraint


class TestEvaluator:
    def test_evaluator(self, evaluation_dataset: EvaluationDataset) -> None:
        tags = evaluation_dataset.tags
        if DatasetTag.HAS_NUMERIC_TRAINING_FEATURES in tags:
            pytest.skip("Skipping test for dataset with numeric training features.")

        dataset = evaluation_dataset.dataset
        evaluator = Evaluator()
        constraints: list[Constraint] = []
        evaluator.evaluate(dataset, constraints)
