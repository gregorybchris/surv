from pathlib import Path

import pytest
from surv.models.dataset import Dataset


@pytest.fixture(scope="session", autouse=True)
def house_dataset(house_dataset_train: Dataset, house_dataset_test: Dataset) -> tuple[Dataset, Dataset]:
    return house_dataset_train, house_dataset_test


@pytest.fixture(scope="session", autouse=True)
def house_dataset_train() -> Dataset:
    data_dirpath = Path(__file__).parent / "data" / "house"
    metadata_filepath = data_dirpath / "metadata.json"
    tabular_filepath = data_dirpath / "tabular-train.csv"
    return Dataset.from_files(tabular_filepath, metadata_filepath)


@pytest.fixture(scope="session", autouse=True)
def house_dataset_test() -> Dataset:
    data_dirpath = Path(__file__).parent / "data" / "house"
    metadata_filepath = data_dirpath / "metadata.json"
    tabular_filepath = data_dirpath / "tabular-test.csv"
    return Dataset.from_files(tabular_filepath, metadata_filepath)
