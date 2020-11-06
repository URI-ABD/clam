from collections import namedtuple
from typing import Union, List, Callable, Tuple

import numpy as np

Data = Union[np.memmap, np.ndarray]  # the entire dataset
Datum = Union[int, np.ndarray]  # a point, or the index of a point
Radius = Union[float, int, np.float64]
Vector = List[int]
DistanceFunc = Callable[[Data, Data], Radius]
Metric = Union[str, DistanceFunc]
