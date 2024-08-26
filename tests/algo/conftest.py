from pathlib import Path

import pytest
from surv.models.dataset import Dataset


@pytest.fixture(scope="session", autouse=True)
def house_dataset() -> Dataset:
    data_dirpath = Path(__file__).parent / "data" / "house"
    metadata_filepath = data_dirpath / "metadata.json"
    tabular_filepath = data_dirpath / "tabular.csv"
    return Dataset.from_files(tabular_filepath, metadata_filepath)
