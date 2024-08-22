from dataclasses import dataclass

import numpy as np


@dataclass
class Id3:
    """ID3 Decision Tree Algorithm."""

    max_depth: int = 3
    min_leaf_size: int = 1

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Fit the model to the data.

        Args:
            x_data (np.ndarray): Features.
            y_data (np.ndarray): Target variable.
        """
        pass

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Predict the target variable.

        Returns:
            np.ndarray: Predicted target variable.
        """
        return np.array([])
