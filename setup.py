import os
import toml

from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as f:
    long_description = f.read()

cargo = toml.load('Cargo.toml')
setup(
    name='pyclam',
    version=cargo['package']['version'],
    packages=['pyclam'],
    url='https://github.com/URI-ABD/clam',
    license='MIT',
    author='; '.join(cargo['package']['authors']),
    author_email='',
    description='Clustered Learning of Approximate Manifolds',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'scipy', 'toml'],
    python_requires='>=3.6',
)
