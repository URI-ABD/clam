import pathlib
import typing

import datatable
import numpy
from scipy.special import erf

from pyclam import dataset
from pyclam import metric
from pyclam import space
from pyclam.utils import constants
from pyclam.utils import helpers

logger = helpers.make_logger(__name__)


class CsvDataset(dataset.Dataset):
    def __init__(
            self,
            path: pathlib.Path,
            name: str,
            labels: typing.Union[str, list[int], numpy.ndarray],
            *,
            full_data: typing.Optional[datatable.Frame] = None,
            non_feature_columns: list[str] = None,
            indices: list[int] = None,
            normalize: bool = True,
            means: numpy.ndarray = None,
            sds: numpy.ndarray = None,
    ):
        self.full_data: datatable.Frame = datatable.fread(str(path)) if full_data is None else full_data
        column_names = set(self.full_data.names)

        self.__path = path
        self.__name = name

        non_feature_columns = non_feature_columns or list()

        if isinstance(labels, str):
            assert labels in column_names, f'label_column "{labels}" not found in set of column names.'
            non_feature_columns.append(labels)
            self.__labels: numpy.ndarray = numpy.asarray(self.full_data[:, labels]).astype(numpy.uint).squeeze()
        else:
            self.__labels: numpy.ndarray = numpy.asarray(labels, dtype=numpy.uint)

        self.__non_feature_columns: list[str] = non_feature_columns or list()
        for nfc in self.__non_feature_columns:
            assert nfc in column_names, f'non_feature_column "{nfc}" not found in set of column names.'

        self.__feature_columns = list(column_names - set(self.__non_feature_columns))
        self.__indices = numpy.asarray(list(range(self.full_data.nrows))) if indices is None else numpy.asarray(indices)
        self.__features: datatable.Frame = self.full_data[:, self.__feature_columns]
        self.__normalize = normalize

        means = self.__features.mean() if means is None else means
        sds = (self.__features.sd() * numpy.sqrt(2) + constants.EPSILON) if sds is None else sds

        fill_kwargs = dict(
            nan=constants.EPSILON,
            posinf=constants.EPSILON,
            neginf=constants.EPSILON,
        )
        self.__means = numpy.nan_to_num(means, **fill_kwargs)
        self.__sds = numpy.nan_to_num(sds, **fill_kwargs)

        self.__shape = self.__indices.shape[0], len(self.__feature_columns)

        logger.info(f'Created CsvDataset {name} with shape {self.__shape}.')

    @property
    def name(self) -> str:
        return self.__name

    @property
    def path(self) -> pathlib.Path:
        return self.__path

    @property
    def data(self) -> datatable.Frame:
        return self.__features

    @property
    def indices(self) -> numpy.ndarray:
        return self.__indices

    @property
    def labels(self) -> numpy.ndarray:
        return self.__labels

    def __eq__(self, other: 'CsvDataset') -> bool:
        return self.__name == other.__name

    @property
    def max_instance_size(self) -> int:
        return 8 * self.data.shape[1]

    @property
    def approx_memory_size(self) -> int:
        return self.cardinality * self.max_instance_size

    def __getitem__(self, item: typing.Union[int, typing.Iterable[int]]):
        indices = self.__indices[item]
        rows = numpy.nan_to_num(
            numpy.asarray(self.data[indices, :]),
            nan=self.__means,
            posinf=self.__means,
            neginf=self.__means,
        )

        if self.__normalize:
            rows = (1 + erf((rows - self.__means) / self.__sds)) / 2

        return rows[0] if isinstance(item, int) else rows

    def subset(self, indices: list[int], subset_name: str, labels=None) -> 'CsvDataset':
        return CsvDataset(
            self.__path,
            subset_name,
            full_data=self.full_data,
            labels=self.__labels if labels is None else labels,
            non_feature_columns=self.__non_feature_columns,
            indices=indices,
            normalize=self.__normalize,
            means=self.__means,
            sds=self.__sds,
        )


class CsvSpace(space.MetricSpace):

    def __init__(self, data: CsvDataset, distance_metric: metric.Metric, use_cache: bool):
        super().__init__(use_cache)
        self.__data = data
        self.__distance_metric = distance_metric

    @property
    def data(self) -> CsvDataset:
        return self.__data

    @property
    def distance_metric(self) -> metric.Metric:
        return self.__distance_metric

    def are_instances_equal(self, left: int, right: int) -> bool:
        return self.distance_one_to_one(left, right) == 0.

    def subset(self, indices: list[int], subset_data_name: str) -> 'CsvSpace':
        return CsvSpace(
            self.data.subset(indices, subset_data_name),
            self.distance_metric,
            self.uses_cache,
        )

    def distance_one_to_one(self, left: int, right: int) -> float:
        return super().distance_one_to_one(left, right)

    def distance_one_to_many(self, left: int, right: list[int]) -> numpy.ndarray:
        return super().distance_one_to_many(left, right)

    def distance_many_to_many(self, left: list[int], right: list[int]) -> numpy.ndarray:
        return super().distance_many_to_many(left, right)

    def distance_pairwise(self, indices: list[int]) -> numpy.ndarray:
        return super().distance_pairwise(indices)
