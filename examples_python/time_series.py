import numpy
from dtaidistance import dtw

from pyclam import metric


class DTWMetric(metric.Metric):
    def __init__(self, **kwargs):
        super().__init__(name='dtw_distance')

        # Use key word arguments for any extra parameters needed when calling
        # dtw.distance
        self.kwargs = kwargs

    def __eq__(self, other: 'DTWMetric') -> bool:
        return self.name == other.name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def one_to_one(self, left, right) -> float:
        """ `left` and `right` are each a single data instance. Compute the
         distance between the two.
        """
        return float(dtw.distance(left, right, **self.kwargs))

    def one_to_many(self, left, right) -> numpy.ndarray:
        """ `left` is a single instance and `right` is multiple instances.
         Compute a 1d array of distances from `left` to instances in `right`.
        """
        distances = [self.one_to_one(left, r) for r in right]
        return numpy.asarray(distances, dtype=numpy.float32)

    def many_to_many(self, left, right) -> numpy.ndarray:
        """ `left` and `right` are both multiple instances. Compute a 2d array
         of distances from each instance in `left` to each instance in `right`.
        """
        distances = [self.one_to_one(l, right) for l in left]
        return numpy.asarray(distances, dtype=numpy.float32)

    def pairwise(self, instances) -> numpy.ndarray:
        """ Compute a 2d array of distances among each pair in `instances`.
        """
        return self.many_to_many(instances, instances)
