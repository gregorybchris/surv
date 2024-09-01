import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pytest
from surv.dataset.dataset import Dataset


class DatasetTag(StrEnum):
    HAS_NUMERIC_TRAINING_FEATURES = "has_numeric_training_features"


@dataclass
class EvaluationDataset:
    dataset: Dataset
    tags: set[DatasetTag]


def get_evaluation_dataset_names() -> list[str]:
    return ["house", "latitude"]


@pytest.fixture(scope="session", params=get_evaluation_dataset_names())
def evaluation_dataset(request: pytest.FixtureRequest) -> EvaluationDataset:
    data_dirpath = Path(__file__).parent / "data" / request.param
    feature_info_filepath = data_dirpath / "features.json"
    tabular_filepath = data_dirpath / "tabular.csv"
    dataset = Dataset.from_files(tabular_filepath, feature_info_filepath)

    dataset.validate()

    tags_filepath = data_dirpath / "tags.json"
    with tags_filepath.open("r") as file:
        tags = json.load(file)

    return EvaluationDataset(dataset, tags)
