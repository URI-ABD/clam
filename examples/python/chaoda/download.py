import pathlib

from . import anomaly_data


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
