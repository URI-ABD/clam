import logging
import os

LOG_LEVEL = getattr(logging, os.environ.get('CLAM_LOG', 'INFO'))

SUBSAMPLE_LIMIT = 100
BATCH_SIZE = 10_000
EPSILON = 1e-8

# This is a hack around type-hinting. https://peps.python.org/pep-0661/
class Unset: pass
UNSET = Unset()
