import argparse
import logging
import os
import pathlib

from pyclam.utils import helpers
from . import download
from . import inference
from . import training

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = helpers.make_logger(__name__)

DATA_ROOT = pathlib.Path(os.environ.get(
    'DATA_ROOT',
    pathlib.Path(__file__).parent.parent.parent.parent.joinpath('data'),
)).resolve()
assert DATA_ROOT.exists(), f'DATA_ROOT not found: {DATA_ROOT}'

DATA_DIR = DATA_ROOT.joinpath('anomaly_data')
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = DATA_ROOT.joinpath('trained_models')
OUTPUT_DIR.mkdir(exist_ok=True)

MODES = [
    'download',
    'training',
    'inference',
]

logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(
    prog='chaoda',
    description='Run CHAODA in its various modes with default settings.',
)

""" Define the arguments """
parser.add_argument(
    '--mode', dest='mode', type=str, required=True,
    help=f'What mode to run. Must be one of {MODES}.',
)

args = parser.parse_args()

mode = args.mode
if mode not in MODES:
    raise ValueError(f'`mode` must be one of {MODES}. Got {mode} instead.')

if mode == 'download':
    download.download_and_save(DATA_DIR)
    download.load(DATA_DIR)
elif mode == 'training':
    models_dir = OUTPUT_DIR.joinpath('pretrained_models')
    models_dir.mkdir(exist_ok=True)
    training.default_training(DATA_DIR, models_dir)
else:  # mode == 'inference'
    inference_dir = OUTPUT_DIR.joinpath('inference_results')
    inference_dir.mkdir(exist_ok=True)
    inference.run_inference(DATA_DIR, inference_dir)
