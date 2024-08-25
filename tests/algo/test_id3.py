from surv.algo.id3 import Id3
from surv.models.dataset import Dataset


class TestId3:
    def test_id3(self, house_dataset: tuple[Dataset, Dataset]) -> None:
        dataset_train, dataset_test = house_dataset
        id3 = Id3()
        id3.fit(dataset_train)
        # dataset_pred = id3.predict(dataset_test)
        # print(dataset_pred)
