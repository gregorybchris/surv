from surv.algo.constraints import Constraint
from surv.algo.evaluator import Evaluator
from surv.models.dataset import Dataset


class TestEvaluator:
    def test_evaluator(self, house_dataset: Dataset) -> None:
        evaluator = Evaluator()
        constraints: list[Constraint] = []
        evaluator.evaluate(house_dataset, constraints)
