import abc

import numpy

from . import anomaly_dataset
from .. import core


class AnomalySpace(core.Space):

    @property
    @abc.abstractmethod
    def data(self) -> anomaly_dataset.AnomalyDataset:
        pass

    @abc.abstractmethod
    def subset(self, indices: list[int], subset_data_name: str) -> 'AnomalySpace':
        """ See the `Dataset.subset`
        """
        pass


class AnomalyTabularSpace(AnomalySpace):

    def __init__(
            self,
            data: anomaly_dataset.AnomalyDataset,
            distance_metric: core.Metric,
            use_cache: bool,
    ):
        self.__data = data
        self.__distance_metric = distance_metric
        super().__init__(use_cache)

    @property
    def data(self) -> anomaly_dataset.AnomalyDataset:
        return self.__data

    def subset(self, indices: list[int], subset_data_name: str) -> 'AnomalyTabularSpace':
        return AnomalyTabularSpace(
            self.data.subset(indices, subset_data_name),
            self.distance_metric,
            self.uses_cache,
        )

    @property
    def distance_metric(self) -> core.Metric:
        return self.__distance_metric

    def are_instances_equal(self, left: int, right: int) -> bool:
        return self.distance_one_to_one(left, right) == 0.

    def distance_one_to_one(self, left: int, right: int) -> float:
        if self.uses_cache:
            return super().distance_one_to_one(left, right)
        else:
            return self.distance_metric.one_to_one(self.data[left], self.data[right])

    def distance_one_to_many(self, left: int, right: list[int]) -> numpy.ndarray:
        if self.uses_cache:
            return super().distance_one_to_many(left, right)
        else:
            return self.distance_metric.one_to_many(self.data[left], self.data[right])

    def distance_many_to_many(self, left: list[int], right: list[int]) -> numpy.ndarray:
        if self.uses_cache:
            return super().distance_many_to_many(left, right)
        else:
            return self.distance_metric.many_to_many(self.data[left], self.data[right])

    def distance_pairwise(self, indices: list[int]) -> numpy.ndarray:
        return self.distance_metric.pairwise(self.data[indices])


__all__ = [
    'AnomalySpace',
    'AnomalyTabularSpace',
]
