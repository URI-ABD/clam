import logging
import os

LOG_LEVEL = getattr(logging, os.environ.get('CLAM_LOG', 'INFO'))

SUBSAMPLE_LIMIT = 100
BATCH_SIZE = 10_000
EPSILON = 1e-8
