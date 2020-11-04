from collections import namedtuple
from typing import Union, List, Callable

import numpy as np

Data = Union[np.memmap, np.ndarray]
Radius = Union[float, int, np.float64]
Vector = List[int]
DistanceFunc = Callable[[Data, Data], Radius]
Metric = Union[str, DistanceFunc]
