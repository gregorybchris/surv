from pathlib import Path

import pytest
from surv.algo.id3 import Id3
from surv.models.dataset import Dataset


@pytest.fixture(scope="session", autouse=True)
def dataset_train() -> Dataset:
    data_dirpath = Path(__file__).parent / "data"
    responses_filepath = data_dirpath / "responses-train.csv"
    questions_filepath = data_dirpath / "questions.json"
    return Dataset.from_files(questions_filepath, responses_filepath)


@pytest.fixture(scope="session", autouse=True)
def dataset_test() -> Dataset:
    data_dirpath = Path(__file__).parent / "data"
    responses_filepath = data_dirpath / "responses-test.csv"
    questions_filepath = data_dirpath / "questions.json"
    return Dataset.from_files(questions_filepath, responses_filepath)


class TestId3:
    def test_id3(self, dataset_train: Dataset, dataset_test: Dataset) -> None:
        id3 = Id3()
        id3.fit(dataset_train)
        dataset_pred = id3.predict(dataset_test)
        print(dataset_pred)
