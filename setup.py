import pathlib

import toml
from setuptools import setup

with open(pathlib.Path(__file__).parent.joinpath('README.md'), 'r') as f:
    long_description = f.read()

cargo = toml.load('Cargo.toml')
setup(
    name='pyclam',
    version=cargo['package']['version'],
    packages=['pyclam', 'pyclam.anomaly_detection', 'pyclam.classification', 'pyclam.core', 'pyclam.search', 'pyclam.utils'],
    url='https://github.com/URI-ABD/clam',
    license='MIT',
    author='; '.join(cargo['package']['authors']),
    author_email='',
    description='Clustered Learning of Approximate Manifolds',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy>=1.22,<1.23', 'scipy>=1.8,<1.9', 'toml>=0.10,<0.11', 'scikit-learn>=1.1,<1.2'],
    python_requires='>=3.9,<3.11',
)
