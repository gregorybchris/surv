import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import TypeAdapter

from surv.models.question import Question


@dataclass
class Dataset:
    """Survey dataset."""

    questions: list[Question]
    responses: pd.DataFrame

    @classmethod
    def from_files(cls, questions_filepath: Path, responses_filepath: Path) -> "Dataset":
        """Load the dataset from the data directory.

        Args:
            questions_filepath (Path): Path to the questions JSON file.
            responses_filepath (Path): Path to the responses CSV file.
        """
        with questions_filepath.open("r") as file:
            questions_json = json.load(file)
            type_adapter = TypeAdapter(list[Question])
            questions = type_adapter.validate_python(questions_json)

        responses = pd.read_csv(responses_filepath)

        return cls(questions, responses)
