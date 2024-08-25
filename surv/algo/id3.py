from dataclasses import dataclass

from surv.models.dataset import Dataset


@dataclass
class Id3:
    """ID3 Decision Tree Algorithm."""

    max_depth: int = 3
    min_leaf_size: int = 1

    def fit(self, dataset: Dataset) -> None:
        """Fit the model to the data.

        Args:
            dataset (Dataset): Dataset to fit the model to.
        """
        raise NotImplementedError

    def predict(self, dataset: Dataset) -> Dataset:
        """Predict the target variable.

        Returns:
            Dataset: Dataset with the predicted target variable.
        """
        raise NotImplementedError
