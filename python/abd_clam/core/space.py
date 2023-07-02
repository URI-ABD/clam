"""This module defines the `Space` class."""

import abc
import random
import typing

import numpy

from . import dataset
from . import metric


class Space(abc.ABC):
    """This class combines a `Dataset` and a `Metric` into a `MetricSpace`.

    We build `Cluster`s and `Graph`s over a `MetricSpace`. This class provides
    access to the underlying `Dataset` and `Metric`. Subclasses should
    implement the methods to compute distances between indexed instances using
    the underlying `Metric` and `Dataset`.

    For those cases where the distance function in `Metric` is expensive to
    compute, this class provides a cache which stores the distance values
    between pairs of instances by their indices. If you want to use the cache,
    set the `use_cache` parameter to True when instantiating a metric space.
    """

    def __init__(self, use_cache: bool) -> None:
        """Initialize the `MetricSpace`."""
        self.__use_cache = use_cache
        self.__cache: dict[tuple[int, int], float] = {}

    @property
    def name(self) -> str:
        """The name of the `MetricSpace`."""
        return f"{self.data.name}__{self.distance_metric.name}"

    @property
    def uses_cache(self) -> bool:
        """Whether the object distance values."""
        return self.__use_cache

    @property
    @abc.abstractmethod
    def data(self) -> dataset.Dataset:
        """The `Dataset` used to compute distances between instances."""
        pass

    @property
    @abc.abstractmethod
    def distance_metric(self) -> metric.Metric:
        """The `Metric` used to compute distances between instances."""
        pass

    @abc.abstractmethod
    def are_instances_equal(self, left: int, right: int) -> bool:
        """Given two indices, returns whether the corresponding instances are equal.

        Usually, this will rely on using a `Metric`. Two equal instances
        should have a distance of zero between them. The default implementation
        relies on this.
        """
        return self.distance_one_to_one(left, right) == 0.0

    @abc.abstractmethod
    def subspace(self, indices: list[int], subset_data_name: str) -> "Space":
        """See the `Dataset.subset`."""
        pass

    @abc.abstractmethod
    def distance_one_to_one(self, left: int, right: int) -> float:
        """See the `Dataset.one_to_one`. The default implementation uses the cache."""
        if not self.is_in_cache(left, right):
            d = self.distance_metric.one_to_one(self.data[left], self.data[right])
            self.add_to_cache(left, right, d)
        return self.get_from_cache(left, right)

    @abc.abstractmethod
    def distance_one_to_many(self, left: int, right: list[int]) -> numpy.ndarray:
        """See the `Dataset.one_to_many`. The default implementation uses the cache."""
        distances = [self.distance_one_to_one(left, r) for r in right]
        return numpy.asarray(distances, dtype=numpy.float32)

    @abc.abstractmethod
    def distance_many_to_many(self, left: list[int], right: list[int]) -> numpy.ndarray:
        """See the `Dataset.many_to_many`. The default implementation uses the cache."""
        distances = [self.distance_one_to_many(i, right) for i in left]
        return numpy.stack(distances)

    @abc.abstractmethod
    def distance_pairwise(self, indices: list[int]) -> numpy.ndarray:
        """See the `Dataset.pairwise`. The default implementation uses the cache."""
        return self.distance_many_to_many(indices, indices)

    @staticmethod
    def __cache_key(i: int, j: int) -> tuple[int, int]:
        """This works because of the `symmetry` property of a `Metric`."""
        return (i, j) if i < j else (j, i)

    def is_in_cache(self, i: int, j: int) -> bool:
        """Checks whether the distance between `i` and `j` is in the cache."""
        return self.__cache_key(i, j) in self.__cache

    def get_from_cache(self, i: int, j: int) -> float:
        """Returns the distance between `i` and `j` from the cache.

        Raises a KeyError if the distance value is not in the cache.
        """
        return self.__cache[self.__cache_key(i, j)]

    def add_to_cache(self, i: int, j: int, distance: float) -> None:
        """Adds the given `distance` to the cache."""
        self.__cache[self.__cache_key(i, j)] = distance

    def remove_from_cache(self, i: int, j: int) -> float:
        """Removes the distance between `i` and `j` from the cache."""
        return self.__cache.pop(  # type: ignore[call-overload]
            self.__cache_key(i, j),
            default=0.0,
        )

    def clear_cache(self) -> int:
        """Empty the cache and return the number of items that were in the cache."""
        num_items = len(self.__cache)
        self.__cache.clear()
        return num_items

    def choose_unique(
        self,
        n: int,
        indices: typing.Optional[list[int]] = None,
    ) -> list[int]:
        """Randomly chooses `n` unique instances from the dataset.

        Args:
            n: The number of unique instances to choose.
            indices: Optional. The list of indices from which to choose.

        Returns:
            A randomly selected list of indices of `n` unique instances.
        """
        indices = indices or list(range(self.data.cardinality))

        if not (0 < n <= len(indices)):
            msg = (
                f"`n` must be a positive integer no larger than the length of "
                f"`indices` ({len(indices)}). Got {n} instead."
            )
            raise ValueError(msg)

        randomized_indices = indices.copy()
        random.shuffle(randomized_indices)

        if n == len(randomized_indices):
            return randomized_indices

        chosen: list[int] = []
        for i in randomized_indices:
            for o in chosen:
                if self.are_instances_equal(i, o):
                    break
            else:
                chosen.append(i)
                if len(chosen) == n:
                    break

        return chosen


class TabularSpace(Space):
    """A `Space` that uses a `TabularDataset` and a `Metric`."""

    def __init__(
        self,
        data: typing.Union[dataset.TabularDataset, dataset.TabularMMap],
        distance_metric: metric.Metric,
        use_cache: bool,
    ) -> None:
        """Initialize the `MetricSpace`."""
        self.__data = data
        self.__distance_metric = distance_metric
        super().__init__(use_cache)

    @property
    def data(
        self,
    ) -> typing.Union[dataset.TabularDataset, dataset.TabularMMap]:
        """Return the data."""
        return self.__data

    @property
    def distance_metric(self) -> metric.Metric:
        """Return the distance metric."""
        return self.__distance_metric

    def are_instances_equal(self, left: int, right: int) -> bool:
        """Return whether the instances at `left` and `right` are identical."""
        return self.distance_one_to_one(left, right) == 0.0

    def distance_one_to_one(self, left: int, right: int) -> float:
        """Compute the distance between `left` and `right`."""
        if self.uses_cache:
            return super().distance_one_to_one(left, right)
        return self.distance_metric.one_to_one(self.data[left], self.data[right])

    def distance_one_to_many(
        self,
        left: int,
        right: list[int],
    ) -> numpy.ndarray:
        """Compute the distances between `left` and each instance in `right`."""
        if self.uses_cache:
            return super().distance_one_to_many(left, right)
        return self.distance_metric.one_to_many(self.data[left], self.data[right])

    def distance_many_to_many(
        self,
        left: list[int],
        right: list[int],
    ) -> numpy.ndarray:
        """Compute the distances between instances in `left` and `right`."""
        if self.uses_cache:
            return super().distance_many_to_many(left, right)
        return self.distance_metric.many_to_many(self.data[left], self.data[right])

    def distance_pairwise(self, indices: list[int]) -> numpy.ndarray:
        """Compute the distances between all pairs of instances in `indices`."""
        return self.distance_metric.pairwise(self.data[indices])

    def subspace(
        self,
        indices: list[int],
        subset_data_name: str,
    ) -> "TabularSpace":
        """Return a subspace of this space."""
        return TabularSpace(
            self.data.subset(indices, subset_data_name),
            self.distance_metric,
            self.uses_cache,
        )


__all__ = [
    "Space",
    "TabularSpace",
]
