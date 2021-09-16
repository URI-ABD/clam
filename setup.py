"""CLAM Python Setup.

This file pulls all important information from elsewhere.
If you are changing something here, you probably went wrong in life earlier.
"""
import os

from setuptools import setup

DIR = os.path.dirname(__file__)

# Get description from the README.
with open(os.path.join(DIR, 'README.md'), 'r') as f:
    long_description = f.read()

# Pull version number from Rust, because Rust is Truth.
with open(os.path.join(DIR, 'Cargo.toml'), 'r') as f:
    contents = f.readlines()

version = list(filter(lambda l: l.startswith('version'), contents))
assert len(version) == 1, f"Multiple versions found! {version}"
version = version[0].split('=')[1].strip()[1:-1]
assert version[0] != '"' and version[-1] != '"', f"Version seems to still have quotes! {version}"
assert len(version.split('.')) == 3, f"Version contains too many parts! {version}"

# Put it all together!
setup(
    name='pyclam',
    version=version,
    packages=['pyclam'],
    url='https://github.com/URI-ABD/clam',
    license='MIT',
    author_email='',
    description='Clustered Learning of Approximate Manifolds',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy==1.*', 'scipy==1.*'],
    python_requires='>=3.6',
)
