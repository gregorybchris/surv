import numpy as np
from surv.algo.id3 import Id3


class TestId3:
    def test_id3(self) -> None:
        id3 = Id3()
        x_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([1, 0])
        id3.fit(x_train, y_train)
        x_test = np.array([[1, 2], [3, 4]])
        y_pred = id3.predict(x_test)
        print(y_pred)
