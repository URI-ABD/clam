import logging
import math
import time
import typing

import h5py
import numpy

from pyclam import cluster_criteria
from pyclam import dataset
from pyclam import metric
from pyclam import space
from pyclam.search import cakes
from pyclam.utils import helpers
from utils import paths

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = helpers.make_logger(__name__)


SEARCH_DATA_DIR = paths.DATA_ROOT.joinpath('search_data')
assert SEARCH_DATA_DIR.exists()


class HDF5Dataset(dataset.Dataset):

    def __init__(self, data: h5py.Dataset, name: str, indices: list[int] = None):
        super().__init__()
        self.__data = data
        self.__name = name
        self.__indices = numpy.asarray(indices or list(range(data.shape[0])))

    @property
    def name(self) -> str:
        return self.__name

    @property
    def data(self) -> h5py.Dataset:
        return self.__data

    def __eq__(self, other: 'HDF5Dataset') -> bool:
        return self.name == other.name

    @property
    def cardinality(self) -> int:
        return len(self.__indices)

    @property
    def max_instance_size(self) -> int:
        num_bytes = self.data.dtype.itemsize
        num_features = self.data.shape[1]
        return num_features * num_bytes

    @property
    def approx_memory_size(self) -> int:
        return self.cardinality * self.max_instance_size

    def __getitem__(self, item: typing.Union[int, typing.Iterable[int]]):
        if isinstance(item, int):
            item = int(self.__indices[item])
            return self.data[item]
        else:
            item = numpy.asarray(item)
            indices = numpy.argsort(item)
            sorted_instances = self.data[item[indices]]
            instances = numpy.zeros_like(sorted_instances)
            instances[indices, :] = sorted_instances

            return instances

    def subset(self, indices: list[int], subset_name: str) -> 'HDF5Dataset':
        return HDF5Dataset(self.data, subset_name, indices)


class HDF5Space(space.Space):

    def __init__(self, data: HDF5Dataset, distance_metric: metric.Metric):
        super().__init__(True)
        self.__data = data
        self.__distance_metric = distance_metric

    @property
    def data(self) -> dataset.Dataset:
        return self.__data

    @property
    def distance_metric(self) -> metric.Metric:
        return self.__distance_metric

    def are_instances_equal(self, left: int, right: int) -> bool:
        return self.distance_one_to_one(left, right) == 0.

    def subset(self, indices: list[int], subset_data_name: str) -> 'HDF5Space':
        return HDF5Space(
            HDF5Dataset(self.data, subset_data_name, indices),
            self.distance_metric
        )

    def distance_one_to_one(self, left: int, right: int) -> float:
        return super().distance_one_to_one(left, right)

    def distance_one_to_many(self, left: int, right: list[int]) -> numpy.ndarray:
        return super().distance_one_to_many(left, right)

    def distance_many_to_many(self, left: list[int], right: list[int]) -> numpy.ndarray:
        return super().distance_many_to_many(left, right)

    def distance_pairwise(self, indices: list[int]) -> numpy.ndarray:
        return super().distance_pairwise(indices)


def bench_sift():
    data_path = SEARCH_DATA_DIR.joinpath('as_hdf5').joinpath('sift.hdf5')

    with h5py.File(data_path, 'r') as reader:
        distance_metric = metric.ScipyMetric(reader.attrs['distance'])
        train_data = HDF5Dataset(reader['train'], 'sift_train')
        test_data = HDF5Dataset(reader['test'], 'sift_test')
        neighbors_data = HDF5Dataset(reader['neighbors'], 'sift_neighbors')
        distances_data = HDF5Dataset(reader['distances'], 'sift_distances')

        train_space = HDF5Space(train_data, distance_metric)

        start = time.perf_counter()
        searcher = cakes.CAKES(train_space).build(
            max_depth=None,
            additional_criteria=[cluster_criteria.MinPoints(int(math.log2(train_data.cardinality)))]
        )
        end = time.perf_counter()
        build_time = end - start

        times = list()
        accuracies = list()
        for i in range(test_data.cardinality):
            logger.info(f'Searching query {i} ...')

            start = time.perf_counter()
            results = searcher.knn_search(test_data[i], k=100)
            end = time.perf_counter()
            times.append(end - start)

            true_hits = set(neighbors_data[i])
            matches = true_hits.intersection(results)
            accuracies.append(len(matches) / 100)

    mean_search_time = sum(times) / len(times)
    mean_search_accuracy = sum(accuracies) / len(accuracies)

    print(f'Building CAKES took {build_time:.2e} seconds with {train_data.cardinality} instances.')
    print(f'Mean search time was {mean_search_time:.2e} seconds.')
    print(f'Mean search accuracy was {mean_search_accuracy:.3f}')

    return distance_metric, train_data, test_data, neighbors_data, distances_data


def convert_to_npy():
    hdf5_root = SEARCH_DATA_DIR.joinpath('as_hdf5')
    npy_root = SEARCH_DATA_DIR.joinpath('as_npy')

    for hdf5_path in hdf5_root.iterdir():
        name = hdf5_path.name.split('.')[0]

        with h5py.File(hdf5_path, 'r') as reader:
            for subset in ['train', 'test', 'neighbors', 'distances']:
                data = numpy.asarray(reader[subset])
                print(f'{name}, {subset} {data.shape}, {data.dtype}')
                numpy.save(npy_root.joinpath(f'{name}_{subset}.npy'), data, allow_pickle=False, fix_imports=False)

    return


if __name__ == '__main__':
    convert_to_npy()
