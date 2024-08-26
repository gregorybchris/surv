from surv.algo.evaluator import Evaluator
from surv.models.dataset import Dataset


class TestEvaluator:
    def test_evaluator(self, house_dataset: Dataset) -> None:
        evaluator = Evaluator()
        evaluator.evaluate(house_dataset)
