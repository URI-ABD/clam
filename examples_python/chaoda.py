import pathlib

import anomaly_data
from utils import paths

DATA_DIR = paths.DATA_ROOT.joinpath('anomaly_data')


def download_and_save(data_dir: pathlib.Path):

    for name, url in sorted(anomaly_data.DATASET_URLS.items()):
        data = anomaly_data.AnomalyData(data_dir, name, url).download().preprocess()
        save_path = data.save()

        print(f'Saved {data.name} to {save_path}')

    return


def load(data_dir: pathlib.Path):

    for name in anomaly_data.DATASET_URLS:
        data = anomaly_data.AnomalyData.load(data_dir, name)

        print(f'loaded {data.name} data with features of shape {data.features.shape}.')

    return


if __name__ == '__main__':
    DATA_DIR.mkdir(exist_ok=True)
    download_and_save(DATA_DIR)
    load(DATA_DIR)
