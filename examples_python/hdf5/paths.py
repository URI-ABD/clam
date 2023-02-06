import os
import pathlib

DATA_ROOT = pathlib.Path(os.environ.get(
    'DATA_ROOT',
    pathlib.Path(__file__).parent.parent.parent.parent.joinpath('data'),
)).resolve()
assert DATA_ROOT.exists(), f'DATA_ROOT not found: {DATA_ROOT}'

REPORTS_DIR = DATA_ROOT.joinpath('reports')
assert REPORTS_DIR.exists(), f'Path not found: {REPORTS_DIR}'
