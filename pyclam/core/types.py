import typing

import numpy

Dataset = numpy.ndarray  # the entire dataset, TODO: Replace with custom DataLoader class
Datum = numpy.ndarray  # a point,
Distance = typing.Union[numpy.float64, float]
Indices = list[int]  # TODO: Rename to Indices of Argpoints
DistanceFunc = typing.Callable[[Datum, Datum], Distance]
Metric = typing.Union[str, DistanceFunc]  # TODO: Compile a list of metrics supported by scipy and by CLAM.
