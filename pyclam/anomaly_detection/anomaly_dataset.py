import abc
import typing

import numpy

from .. import core


class AnomalyDataset(core.Dataset):

    @property
    @abc.abstractmethod
    def labels(self) -> numpy.ndarray:
        pass


class AnomalyTabular(AnomalyDataset):

    def __init__(self, data: numpy.ndarray, scores: numpy.ndarray, name: str):
        self.__name = name
        self.__data = data
        self.__scores = scores
        self.__indices = numpy.asarray(list(range(data.shape[0])), dtype=numpy.uint)

    @property
    def name(self) -> str:
        return self.__name

    @property
    def data(self) -> numpy.ndarray:
        return self.__data

    @property
    def labels(self) -> numpy.ndarray:
        return self.__scores

    @property
    def indices(self) -> numpy.ndarray:
        return self.__indices

    def __eq__(self, other: 'AnomalyTabular') -> bool:
        return self.__name == other.__name

    @property
    def max_instance_size(self) -> int:
        item_size = self.__data.itemsize
        num_items = numpy.prod(self.__data.shape[1:])
        return int(item_size * num_items)

    @property
    def approx_memory_size(self) -> int:
        data_size = self.cardinality * self.max_instance_size
        scores_size = self.cardinality * self.__scores.itemsize
        return data_size + scores_size

    def __getitem__(self, item: typing.Union[int, typing.Iterable[int]]) -> numpy.ndarray:
        return self.__data[item]

    def subset(self, indices: list[int], subset_name: str) -> 'AnomalyTabular':
        data = self.__data[indices, :]
        scores = self.__scores[indices]
        return AnomalyTabular(data, scores, subset_name)


__all__ = [
    'AnomalyDataset',
    'AnomalyTabular',
]
