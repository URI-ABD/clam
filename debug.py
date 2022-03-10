import pathlib

import numpy


NUM_OUTLIERS = 5
MULTIPLIER = 9
NUM_INLIERS = MULTIPLIER * NUM_OUTLIERS
DATA_ROOT = pathlib.Path(__file__).parent.joinpath('data')
SOURCE_DATA = 'annthyroid'


def create_dummy_dataset():
    data = numpy.load(str(DATA_ROOT.joinpath(f'{SOURCE_DATA}.npy')))
    labels = numpy.load(str(DATA_ROOT.joinpath(f'{SOURCE_DATA}_labels.npy')))

    outlier_indices = numpy.argwhere(labels).flatten()
    sub_outliers = data[outlier_indices[:NUM_OUTLIERS], :]

    inlier_indices = numpy.argwhere(labels == 0).flatten()
    sub_inliers = data[inlier_indices[:NUM_INLIERS], :]

    subset = numpy.concatenate((sub_outliers, sub_inliers), axis=0).astype(data.dtype)
    labels = numpy.asarray([1] * NUM_OUTLIERS + [0] * NUM_INLIERS, dtype=labels.dtype)

    numpy.save(str(DATA_ROOT.joinpath(f'subset.npy')), subset, allow_pickle=False, fix_imports=False)
    numpy.save(str(DATA_ROOT.joinpath(f'subset_labels.npy')), labels, allow_pickle=False, fix_imports=False)
    return


if __name__ == "__main__":
    # print(DATA_ROOT)
    create_dummy_dataset()
