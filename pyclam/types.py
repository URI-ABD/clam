from typing import Callable
from typing import List
from typing import Union

import numpy as np

Data = Union[np.memmap, np.ndarray]  # the entire dataset, TODO: Replace with custom DataLoader class
Datum = Union[int, np.ndarray]  # a point, or the index of a point, TODO: Use only np.array as Datum
Radius = Union[float, int, np.float64]
Vector = List[int]  # TODO: Rename to Indices of Argpoints
DistanceFunc = Callable[[Data, Data], Radius]
Metric = Union[str, DistanceFunc]  # TODO: Compile a list of metrics supported by scipy and by CLAM.
