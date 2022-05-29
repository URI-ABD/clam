import logging

from . import anomaly_detection
from . import classification
from . import search
from . import utils
from .core import criterion
from .core import types
from .core.manifold import Cluster
from .core.manifold import Edge
from .core.manifold import Graph
from .core.manifold import Manifold

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt='%d-%b-%y %H:%M:%S',
)
