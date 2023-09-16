"""CLAM: Clustered Learning of Approximate Manifolds."""

from . import anomaly_detection
from . import classification
from . import core
from . import search
from . import utils
from .core import Cluster
from .core import ClusterCriterion
from .core import Dataset
from .core import Edge
from .core import Graph
from .core import GraphCriterion
from .core import Metric
from .core import Space
from .core import cluster
from .core import cluster_criteria
from .core import dataset
from .core import graph
from .core import graph_criteria
from .core import metric
from .core import space

__version__ = "0.21.6"
