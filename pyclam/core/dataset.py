import abc
import pathlib
import random
import typing

import numpy


class Dataset(abc.ABC):
    """ A `Dataset` is a collection of `instance`s. This abstract class provides
    utilities for accessing instances from the data. A `Dataset` should be
    combined with a `Metric` to produce a `MetricSpace`.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """ Ideally, a user would supply a unique name for each `Dataset` they
    instantiate.
        """
        pass

    @property
    @abc.abstractmethod
    def data(self):
        """ Returns the underlying data.
        """
        pass

    @property
    @abc.abstractmethod
    def indices(self) -> numpy.ndarray:
        pass

    @abc.abstractmethod
    def __eq__(self, other: 'Dataset') -> bool:
        """ Ideally, this would be a quick check based on `name`.
        """
        pass

    @property
    def cardinality(self) -> int:
        """ The number of instances in the data.
        """
        return self.indices.shape[0]

    @property
    @abc.abstractmethod
    def max_instance_size(self) -> int:
        """ The maximum memory size (in bytes) of any instance in the data.
        """
        pass

    @property
    @abc.abstractmethod
    def approx_memory_size(self) -> int:
        """ Returns the approximate memory size (in bytes) of the data.
        """
        pass

    @abc.abstractmethod
    def __getitem__(self, item: typing.Union[int, typing.Iterable[int]]):
        pass

    @abc.abstractmethod
    def subset(self, indices: list[int], subset_name: str) -> 'Dataset':
        """ Returns a subset of the data using only the indexed instances. This
        subset should be of the same class as object from which it is created.
        """
        pass

    def max_batch_size(self, available_memory: int) -> int:
        """ Returns a conservative estimate of the number of instances that will
        fill `available_memory` bytes.

        Args:
            available_memory: in bytes.

        Returns:
            The number of instances that will fill `available_memory`.
        """
        return available_memory // self.max_instance_size

    def complement_indices(self, indices: list[int]) -> list[int]:
        """ Returns the list of indices in the dataset that are not in the given
        `indices`.
        """
        return list(set(range(self.cardinality)) - set(indices))

    def subsample_indices(self, n: int) -> tuple[list[int], list[int]]:
        """ Randomly subsample n indices from the dataset.

        Args:
            n: Number of indices to select.

        Returns:
            A 2-tuple of:
                - list of selected indices.
                - list of remaining indices.
        """
        if not (0 < n < self.cardinality):
            raise ValueError(f'`n` must be a positive integer smaller than the cardinality of the dataset ({self.cardinality}). Got {n} instead.')

        # TODO: This should be from self.indices
        indices = list(range(n))
        random.shuffle(indices)

        return indices[:n], indices[n:]


class TabularDataset(Dataset):
    """ This wraps a 2d numpy array whose rows are instances and columns are
    features.

    To check if two `TabularDataset`s are equal, this class only checks that
    they have the same `name`.

    To check if two instances are equal, this class simply checks for
    element-wise equality among the instances.
    """

    def __init__(self, data: numpy.ndarray, name: str):
        """
        Args:
            data: A 2d array whose rows are instances and columns are features.
            name: This should be unique for each object the user instantiates.
        """
        self.__data = data
        self.__name = name
        self.__indices = numpy.asarray(list(range(data.shape[0])), dtype=numpy.uint)

    @property
    def name(self) -> str:
        return self.__name

    @property
    def data(self) -> numpy.ndarray:
        return self.__data

    @property
    def indices(self) -> numpy.ndarray:
        return self.__indices

    def __eq__(self, other: 'TabularDataset') -> bool:
        return self.__name == other.__name

    @property
    def max_instance_size(self) -> int:
        item_size = self.__data.itemsize
        num_items = numpy.prod(self.__data.shape[1:])
        return int(item_size * num_items)

    @property
    def approx_memory_size(self) -> int:
        return self.cardinality * self.max_instance_size

    def __getitem__(self, item: typing.Union[int, typing.Iterable[int]]) -> numpy.ndarray:
        return self.__data[item]

    def subset(self, indices: list[int], subset_name: str) -> 'TabularDataset':
        return TabularDataset(self.__data[indices, :], subset_name)


class TabularMMap(TabularDataset):
    """ This is identical to a `TabularDataset` but the underlying data is a
    numpy array loaded with `mmap_mode = 'r'`.
    """

    def __init__(
            self,
            file_path: typing.Union[pathlib.Path, str],
            name: str,
            indices: list[int] = None,
    ):
        self.file_path = file_path
        data = numpy.load(str(file_path), mmap_mode='r')
        indices = list(range(data.shape[0])) if indices is None else indices
        self.__indices = numpy.asarray(indices, dtype=numpy.uint)
        if self.__indices.max(initial=0) >= data.shape[0]:
            raise IndexError("Invalid indices provided.")
        
        super().__init__(data, name)

    @property
    def indices(self) -> numpy.ndarray:
        return self.__indices

    def __getitem__(self, item: typing.Union[int, typing.Iterable[int]]):
        indices = self.__indices[item]
        return numpy.asarray(self.data[indices, :])

    def subset(self, indices: list[int], subset_name: str) -> 'TabularDataset':
        return TabularMMap(self.file_path, subset_name, indices)

__all__ = [
    'Dataset',
    'TabularDataset',
    'TabularMMap',
]
