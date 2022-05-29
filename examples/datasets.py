import os
import pathlib

from pyclam.anomaly_detection import datasets

DATA_ROOT = pathlib.Path(os.environ.get(
    'DATA_ROOT',
    pathlib.Path(__file__).parent.parent.parent.joinpath('data'),
)).resolve()


def download_and_save():
    data_dir = DATA_ROOT.joinpath('chaoda_data')
    data_dir.mkdir(exist_ok=True)

    for name, url in sorted(datasets.DATASET_URLS.items()):
        data = datasets.ChaodaData(data_dir, name, url)
        data.download()
        data.preprocess()
        save_path = data.save()

        print(f'Saved {data.name} to {save_path}')

    return


def load():
    data_dir = DATA_ROOT.joinpath('chaoda_data')

    for name in datasets.DATASET_URLS:
        data = datasets.ChaodaData.load(data_dir, name)

        print(f'loaded {data.name} data with features of shape {data.features.shape}.')

    return


if __name__ == '__main__':
    assert DATA_ROOT.exists(), f'DATA_ROOT not found: {DATA_ROOT}'
    download_and_save()
    load()
