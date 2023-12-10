"""This module defines the `Dataset` class."""

import abc
import pathlib
import random
import typing

import numpy


class Dataset(abc.ABC):
    """A `Dataset` is a collection of `instance`s.

    This abstract class provides utilities for accessing instances from the data.
    A `Dataset` should be combined with a `Metric` to produce a `Space`.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Ideally, a user would supply a unique name for each `Dataset`."""
        pass

    @property
    @abc.abstractmethod
    def data(self) -> typing.Any:  # noqa: ANN401
        """Returns the underlying data."""
        pass

    @property
    @abc.abstractmethod
    def indices(self) -> numpy.ndarray:
        """Returns the indices of the instances in the data."""
        pass

    @abc.abstractmethod
    def __eq__(self, other: "Dataset") -> bool:  # type: ignore[override]
        """Ideally, this would be a quick check based on `name`."""
        pass

    @property
    def cardinality(self) -> int:
        """The number of instances in the data."""
        return self.indices.shape[0]

    @property
    @abc.abstractmethod
    def max_instance_size(self) -> int:
        """The maximum memory size (in bytes) of any instance in the data."""
        pass

    @property
    @abc.abstractmethod
    def approx_memory_size(self) -> int:
        """Returns the approximate memory size (in bytes) of the data."""
        pass

    @abc.abstractmethod
    def __getitem__(
        self,
        item: typing.Union[int, typing.Iterable[int]],
    ) -> typing.Any:  # noqa: ANN401
        """Returns the instance(s) at the given index/indices."""
        pass

    @abc.abstractmethod
    def subset(
        self,
        indices: list[int],
        subset_name: str,
    ) -> "Dataset":
        """Returns a subset of the data using only the indexed instances.

        This subset should be of the same class as object from which it is created.
        """
        pass

    def max_batch_size(self, available_memory: int) -> int:
        """Conservatively estimate of the number of instances that will fill memory.

        Args:
            available_memory: in bytes.

        Returns:
            The number of instances that will fill `available_memory`.
        """
        return available_memory // self.max_instance_size

    def complement_indices(self, indices: list[int]) -> list[int]:
        """Returns the indices in the dataset that are not in the given `indices`."""
        return list(set(range(self.cardinality)) - set(indices))

    def subsample_indices(self, n: int) -> tuple[list[int], list[int]]:
        """Randomly subsample n indices from the dataset.

        Args:
            n: Number of indices to select.

        Returns:
            A 2-tuple of:
                - list of selected indices.
                - list of remaining indices.
        """
        if not (0 < n < self.cardinality):
            msg = (
                f"`n` must be a positive integer smaller than the cardinality of "
                f"the dataset ({self.cardinality}). Got {n} instead."
            )
            raise ValueError(msg)

        # TODO: This should be from self.indices
        indices = list(range(n))
        random.shuffle(indices)

        return indices[:n], indices[n:]


class TabularDataset(Dataset):
    """This wraps a 2d numpy array whose rows are instances and columns are features.

    To check if two `TabularDataset`s are equal, this class only checks that
    they have the same `name`.

    To check if two instances are equal, this class simply checks for
    element-wise equality among the instances.
    """

    def __init__(self, data: numpy.ndarray, name: str) -> None:
        """Initializes a `TabularDataset`.

        Args:
            data: A 2d array whose rows are instances and columns are features.
            name: This should be unique for each object the user instantiates.
        """
        self.__data = data
        self.__name = name
        self.__indices = numpy.asarray(list(range(data.shape[0])), dtype=numpy.uint)

    @property
    def name(self) -> str:
        """Returns the name of the dataset."""
        return self.__name

    @property
    def data(self) -> numpy.ndarray:
        """Returns the underlying data."""
        return self.__data

    @property
    def indices(self) -> numpy.ndarray:
        """Returns the indices of the instances in the data."""
        return self.__indices

    def __eq__(self, other: "TabularDataset") -> bool:  # type: ignore[override]
        """Checks if two `TabularDataset`s have the same name."""
        return self.__name == other.__name

    @property
    def max_instance_size(self) -> int:
        """Returns the maximum memory size (in bytes) of any instance in the data."""
        item_size = self.__data.itemsize
        num_items = numpy.prod(self.__data.shape[1:])
        return int(item_size * num_items)

    @property
    def approx_memory_size(self) -> int:
        """Returns the approximate memory size (in bytes) of the data."""
        return self.cardinality * self.max_instance_size

    def __getitem__(
        self,
        item: typing.Union[int, typing.Iterable[int]],
    ) -> numpy.ndarray:
        """Returns the instance(s) at the given index/indices."""
        return self.__data[item]

    def subset(self, indices: list[int], subset_name: str) -> "TabularDataset":
        """Returns a subset of the data using only the indexed instances."""
        return TabularDataset(self.__data[indices, :], subset_name)


class TabularMMap(TabularDataset):
    """This is identical to a `TabularDataset` but the data is a `numpy.memmap`."""

    def __init__(
        self,
        file_path: typing.Union[pathlib.Path, str],
        name: str,
        indices: typing.Optional[list[int]] = None,
    ) -> None:
        """Initializes a `TabularMMap`."""
        self.file_path = file_path
        data = numpy.load(str(file_path), mmap_mode="r")
        indices = list(range(data.shape[0])) if indices is None else indices
        self.__indices = numpy.asarray(indices, dtype=numpy.uint)
        if self.__indices.max(initial=0) >= data.shape[0]:
            msg = "Invalid indices provided."
            raise IndexError(msg)

        super().__init__(data, name)

    @property
    def indices(self) -> numpy.ndarray:
        """Returns the indices of the instances in the data."""
        return self.__indices

    def __getitem__(
        self,
        item: typing.Union[int, typing.Iterable[int]],
    ) -> numpy.ndarray:
        """Returns the instance(s) at the given index/indices."""
        indices = self.__indices[item]
        return numpy.asarray(self.data[indices, :])

    def subset(self, indices: list[int], subset_name: str) -> "TabularMMap":
        """Returns a subset of the data using only the indexed instances."""
        return TabularMMap(  # type: ignore[return-value]
            self.file_path,
            subset_name,
            indices,
        )


__all__ = [
    "Dataset",
    "TabularDataset",
    "TabularMMap",
]
