import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import TypeAdapter

from surv.models.question import Question


@dataclass
class Dataset:
    """Survey dataset."""

    dataframe: pd.DataFrame
    questions: list[Question]

    @classmethod
    def load(cls, data_dirpath: Path) -> "Dataset":
        """Load the dataset from the data directory.

        Args:
            data_dirpath (Path): Data directory path.
        """
        dataset_filepath = data_dirpath / "dataset.csv"
        dataframe = pd.read_csv(dataset_filepath)

        questions_filepath = data_dirpath / "questions.json"
        with questions_filepath.open("r") as file:
            questions_json = json.load(file)
            type_adapter = TypeAdapter(list[Question])
            questions = type_adapter.validate_python(questions_json)

        return cls(dataframe, questions)
