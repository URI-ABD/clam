import logging
import pathlib
import re
import time
import typing

import datatable
import numpy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import csv_space
from pyclam import metric
from pyclam.classification import classifier
from pyclam.core import cluster_criteria
from utils import nyxus_metadata

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt='%d-%b-%y %H:%M:%S',
)


class NyxusDataset(csv_space.CsvDataset):

    def __init__(
            self,
            path: pathlib.Path,
            name: str,
            *,
            file_names_column: str,
            class_labels: typing.Literal['platform', 'tissue'],
            non_feature_columns: list[str] = None,
            indices: list[int] = None,
            normalize: bool = True,
            means: numpy.ndarray = None,
            sds: numpy.ndarray = None,
    ):
        data: datatable.Frame = datatable.fread(str(path))

        assert file_names_column in data.names, f'Column {file_names_column} not found in the dataset.'

        file_names: list[str] = data[file_names_column].to_list()[0]
        pattern = re.compile("(?P<d>\w+)_p(?P<p>\d+)_y(?P<y>\d+)_r(?P<r>\d+)_c(?P<c>\d+).tif")
        matches = list(map(pattern.match, file_names))

        if class_labels == 'tissue':
            labels = list(map(int, (r.group("y") for r in matches)))
        else:
            labels = list(map(int, (r.group("p") for r in matches)))

        super().__init__(
            path=path,
            name=name,
            full_data=data,
            labels=labels,
            non_feature_columns=non_feature_columns,
            indices=indices,
            normalize=normalize,
            means=means,
            sds=sds,
        )


def read_nyxus_cells():
    return NyxusDataset(
        path=nyxus_metadata.CELLS_CSV_PATH,
        name='nyxus_cells',
        file_names_column='mask_image',
        class_labels='tissue',
        non_feature_columns=nyxus_metadata.CELLS_NON_FEATURES,
    )


def read_nyxus_images():
    return NyxusDataset(
        path=nyxus_metadata.IMAGES_CSV_PATH,
        name='nyxus_images',
        file_names_column='mask_image',
        class_labels='tissue',
        non_feature_columns=nyxus_metadata.IMAGES_NON_FEATURES,
    )


# noinspection DuplicatedCode
def experiment():
    nyxus_full = read_nyxus_images()

    test_size = 0.2 if nyxus_full.cardinality <= 50_000 else 10_000
    train_indices, test_indices, train_labels, test_labels = train_test_split(
        nyxus_full.indices,
        nyxus_full.labels,
        test_size=test_size,
        random_state=42,
        stratify=nyxus_full.labels,
    )

    nyxus_train = nyxus_full.subset(train_indices, f'{nyxus_full.name}_train', labels=train_labels)
    nyxus_test = nyxus_full.subset(test_indices, f'{nyxus_full.name}_test', labels=test_labels)

    nyxus_spaces = [
        csv_space.CsvSpace(nyxus_train, metric.ScipyMetric('euclidean'), False),
        csv_space.CsvSpace(nyxus_train, metric.ScipyMetric('cityblock'), False),
    ]

    start = time.perf_counter()
    nyxus_classifier = classifier.Classifier(
        nyxus_train.labels,
        nyxus_spaces,
        partition_criteria=[cluster_criteria.MaxDepth(20)]
    ).build()
    end = time.perf_counter()
    build_time = end - start

    start = time.perf_counter()
    predicted_labels, _ = nyxus_classifier.predict(nyxus_test)
    end = time.perf_counter()
    prediction_time = end - start

    score = accuracy_score(nyxus_test.labels, predicted_labels)

    # 21:47:03  22:36:14

    print(f'Building the {nyxus_full.name} classifier for:')
    print(f'\t{nyxus_train.cardinality} instances and')
    print(f'\t{nyxus_classifier.unique_labels} unique labels')
    print(f'\ttook {build_time:.2e} seconds.')

    print(f'Predicting from the classifier for:')
    print(f'\t{nyxus_test.cardinality} instances took')
    print(f'\ttook {prediction_time:.2e} seconds.')

    print(f'The accuracy score was {score:.3f}')

    return


if __name__ == '__main__':
    experiment()
