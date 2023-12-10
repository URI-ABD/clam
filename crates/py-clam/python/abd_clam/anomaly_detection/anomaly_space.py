"""Specialization of `Space` for anomaly detection."""

import abc

import numpy

from .. import core
from . import anomaly_dataset


class AnomalySpace(core.Space):
    """Specialization of `Space` for anomaly detection."""

    @property
    @abc.abstractmethod
    def data(self) -> anomaly_dataset.AnomalyDataset:
        """See the underlying `AnomalyDataset`."""
        pass

    @abc.abstractmethod
    def subspace(self, indices: list[int], subset_data_name: str) -> "AnomalySpace":
        """Return a subspace of the space."""
        pass


class AnomalyTabularSpace(AnomalySpace):
    """Example implementation of `AnomalySpace` for tabular datasets."""

    def __init__(
        self,
        data: anomaly_dataset.AnomalyDataset,
        distance_metric: core.Metric,
        use_cache: bool,
    ) -> None:
        """Initialize an `AnomalyTabularSpace`."""
        self.__data = data
        self.__distance_metric = distance_metric
        super().__init__(use_cache)

    @property
    def data(self) -> anomaly_dataset.AnomalyDataset:
        """See the underlying `AnomalyDataset`."""
        return self.__data

    def subspace(
        self,
        indices: list[int],
        subset_data_name: str,
    ) -> "AnomalyTabularSpace":
        """Return a subspace of the space."""
        return AnomalyTabularSpace(
            self.data.subset(indices, subset_data_name),  # type: ignore[arg-type]
            self.distance_metric,
            self.uses_cache,
        )

    @property
    def distance_metric(self) -> core.Metric:
        """Distance metric used to compare instances in the space."""
        return self.__distance_metric

    def are_instances_equal(self, left: int, right: int) -> bool:
        """Check if two instances are identical."""
        return self.distance_one_to_one(left, right) == 0.0

    def distance_one_to_one(self, left: int, right: int) -> float:
        """Compute the distance between two instances."""
        if self.uses_cache:
            return super().distance_one_to_one(left, right)
        return self.distance_metric.one_to_one(self.data[left], self.data[right])

    def distance_one_to_many(self, left: int, right: list[int]) -> numpy.ndarray:
        """Compute the distance between one instance and many instances."""
        if self.uses_cache:
            return super().distance_one_to_many(left, right)
        return self.distance_metric.one_to_many(self.data[left], self.data[right])

    def distance_many_to_many(self, left: list[int], right: list[int]) -> numpy.ndarray:
        """Compute the distance between many instances and many instances."""
        if self.uses_cache:
            return super().distance_many_to_many(left, right)
        return self.distance_metric.many_to_many(self.data[left], self.data[right])

    def distance_pairwise(self, indices: list[int]) -> numpy.ndarray:
        """Compute the distance between all pairs of instances."""
        return self.distance_metric.pairwise(self.data[indices])


__all__ = [
    "AnomalySpace",
    "AnomalyTabularSpace",
]
