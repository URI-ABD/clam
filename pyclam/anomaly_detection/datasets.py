import json
import logging
import pathlib
import subprocess
import typing

import h5py
import numpy
import scipy.io

from ..utils import constants
from ..utils import helpers

__all__ = ['DATASET_URLS', 'ChaodaData']

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOG_LEVEL)

MAT_FILE_URLS: dict[str, str] = {
    'annthyroid': 'https://www.dropbox.com/s/aifk51owxbogwav/annthyroid.mat?dl=0',
    'arrhythmia': 'https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=0',
    'breastw': 'https://www.dropbox.com/s/g3hlnucj71kfvq4/breastw.mat?dl=0',
    'cardio': 'https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=0',
    'cover': 'https://www.dropbox.com/s/awx8iuzbu8dkxf1/cover.mat?dl=0',
    'glass': 'https://www.dropbox.com/s/iq3hjxw77gpbl7u/glass.mat?dl=0',
    'ionosphere': 'https://www.dropbox.com/s/lpn4z73fico4uup/ionosphere.mat?dl=0',
    'lympho': 'https://www.dropbox.com/s/ag469ssk0lmctco/lympho.mat?dl=0',
    'mammography': 'https://www.dropbox.com/s/tq2v4hhwyv17hlk/mammography.mat?dl=0',
    'mnist': 'https://www.dropbox.com/s/n3wurjt8v9qi6nc/mnist.mat?dl=0',
    'musk': 'https://www.dropbox.com/s/we6aqhb0m38i60t/musk.mat?dl=0',
    'optdigits': 'https://www.dropbox.com/s/w52ndgz5k75s514/optdigits.mat?dl=0',
    'pendigits': 'https://www.dropbox.com/s/1x8rzb4a0lia6t1/pendigits.mat?dl=0',
    'pima': 'https://www.dropbox.com/s/mvlwu7p0nyk2a2r/pima.mat?dl=0',
    'satellite': 'https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=0',
    'satimage-2': 'https://www.dropbox.com/s/hckgvu9m6fs441p/satimage-2.mat?dl=0',
    'shuttle': 'https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=0',
    'thyroid': 'https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=0',
    'vertebral': 'https://www.dropbox.com/s/5kuqb387sgvwmrb/vertebral.mat?dl=0',
    'vowels': 'https://www.dropbox.com/s/pa26odoq6atq9vx/vowels.mat?dl=0',
    'wbc': 'https://www.dropbox.com/s/ebz9v9kdnvykzcb/wbc.mat?dl=0',
    'wine': 'https://www.dropbox.com/s/uvjaudt2uto7zal/wine.mat?dl=0',
}

HDF5_FILE_URLS = {
    'http': 'https://www.dropbox.com/s/iy9ucsifal754tp/http.mat?dl=0',
    'smtp': 'https://www.dropbox.com/s/dbv2u4830xri7og/smtp.mat?dl=0',
}

DATASET_URLS = {**MAT_FILE_URLS, **HDF5_FILE_URLS}

MMapMode = typing.Literal["r", None]


class ChaodaData:

    __slots__ = [
        'data_dir',
        'name',
        'url',
        'mmap_threshold',
        'mmap_mode',
        'raw_path',
        'features_path',
        'normalized_features_path',
        'scores_path',
    ]

    def __init__(self, data_dir: pathlib.Path, name: str, url: str, mmap_threshold: int = 1024 ** 3, mmap_mode: MMapMode = None):
        self.data_dir = data_dir
        self.name = name
        self.url = url
        self.mmap_threshold = mmap_threshold
        self.mmap_mode: MMapMode = mmap_mode

        data_dir.mkdir(exist_ok=True)
        raw_dir = data_dir.joinpath('raw')
        raw_dir.mkdir(exist_ok=True)

        preprocessed_dir = data_dir.joinpath('preprocessed')
        preprocessed_dir.mkdir(exist_ok=True)

        self.raw_path = raw_dir.joinpath(f'{name}.mat')
        self.features_path = preprocessed_dir.joinpath(f'{name}_features.npy')
        self.normalized_features_path = preprocessed_dir.joinpath(f'{name}_features_normalized.npy')
        self.scores_path = preprocessed_dir.joinpath(f'{name}_scores.npy')

    def save(self) -> pathlib.Path:
        save_path = self.data_dir.joinpath('classes').joinpath(f'{self.name}.json')
        save_path.parent.mkdir(exist_ok=True)
        attributes = {
            'name': self.name,
            'url': self.url,
            'mmap_threshold': self.mmap_threshold,
            'mmap_mode': self.mmap_mode,
            'raw_path': self.raw_path,
            'features_path': self.features_path,
            'normalized_features_path': self.normalized_features_path,
            'scores_path': self.scores_path,
        }

        for k, v in attributes.items():
            if isinstance(v, pathlib.Path):
                relative_path = v.relative_to(self.data_dir)
                attributes[k] = str(relative_path)

        with open(save_path, 'w') as writer:
            json.dump(attributes, writer, indent=4)
        return save_path

    @staticmethod
    def load(data_dir: pathlib.Path, name: str) -> 'ChaodaData':
        save_path = data_dir.joinpath('classes').joinpath(f'{name}.json')
        with open(save_path, 'r') as reader:
            attributes = json.load(reader)

        data = ChaodaData(
            data_dir=data_dir,
            name=attributes['name'],
            url=attributes['url'],
            mmap_threshold=int(attributes['mmap_threshold']),
            mmap_mode=attributes['mmap_mode'],
        )
        for k in ChaodaData.__slots__:
            if 'path' in k:
                v = data_dir.joinpath(pathlib.Path(attributes[k]))
                setattr(data, k, v)

        return data

    def download(
            self,
            extension: str = 'mat',
            force: bool = False,
            suppress_stdout: bool = True,
            suppress_stderr: bool = True,
    ):
        self.raw_path = self.raw_path.with_name(f'{self.name}.{extension}')

        if not force and self.raw_path.exists():
            return

        self.raw_path.unlink(missing_ok=True)

        kwargs = dict()
        if suppress_stdout:
            kwargs['stdout'] = subprocess.DEVNULL
        if suppress_stderr:
            kwargs['stderr'] = subprocess.DEVNULL
        # wget.download(self.url, out=str(self.raw_path))
        subprocess.run(['wget', self.url, '-O', self.raw_path], **kwargs)

        size = self.raw_path.stat().st_size
        if size < 1024:
            warning = f'Downloaded file for {self.name} was too small. {size} bytes.'
            logging.warning(warning)
            raise BytesWarning(warning)

        if size > self.mmap_threshold:
            self.mmap_mode = 'r'

        return

    def preprocess(self, normalization_mode: str = 'gaussian'):
        helpers.catch_normalization_mode(normalization_mode)

        data_dict = dict()
        if self.name in HDF5_FILE_URLS:
            with h5py.File(self.raw_path, 'r') as reader:
                data_dict['X'] = numpy.asarray(reader['X']).T
                data_dict['y'] = numpy.asarray(reader['y']).T
        else:
            data_dict = scipy.io.loadmat(str(self.raw_path))

        features = numpy.asarray(data_dict['X'], dtype=numpy.float32)
        numpy.save(str(self.features_path), features, allow_pickle=False, fix_imports=False)

        features = helpers.normalize(features, mode=normalization_mode)
        numpy.save(str(self.normalized_features_path), features, allow_pickle=False, fix_imports=False)

        scores = numpy.asarray(data_dict['y'], dtype=numpy.uint8).squeeze()
        numpy.save(str(self.scores_path), scores, allow_pickle=False, fix_imports=False)

        return

    @property
    def features(self):
        if not self.features_path.exists():
            raise ValueError(f'Dataset {self.name} as not yet been downloaded.')
        else:
            return numpy.load(str(self.features_path), self.mmap_mode)

    @property
    def scores(self):
        if not self.features_path.exists():
            raise ValueError(f'Dataset {self.name} as not yet been downloaded.')
        else:
            return numpy.load(str(self.scores_path))
