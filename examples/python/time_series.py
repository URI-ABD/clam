import numpy
from dtaidistance import dtw

from abd_clam import metric


class DTWMetric(metric.Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(name="dtw_distance")

        # Use key word arguments for any extra parameters needed when calling
        # dtw.distance
        self.kwargs = kwargs
        self.kwargs["use_c"] = True

    def __eq__(self, other: "DTWMetric") -> bool:
        return self.name == other.name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def one_to_one(self, left: numpy.ndarray, right: numpy.ndarray) -> float:
        """`left` and `right` are each a single time-series. Compute the dtw
        distance between the two.
        """
        return float(dtw.distance(left, right, **self.kwargs))

    def one_to_many(self, left: numpy.ndarray, right: numpy.ndarray) -> numpy.ndarray:
        """`left` is a single time-series and `right` is a 2d array of
        time-series. Compute a 1d array of dtw distances from `left` to each
        time-series in `right`.
        """
        left = left[None, :]
        return self.many_to_many(left, right)[0]

    def many_to_many(self, left: numpy.ndarray, right: numpy.ndarray) -> numpy.ndarray:
        """`left` and `right` are both 2d arrays of time-series. Each row is a
        single time-series.Compute a 2d array of distances from each time-series
        in `left` to each instance in `right`.
        """
        num_left, num_right = left.shape[0], right.shape[0]
        instances = numpy.concatenate([left, right], axis=0)
        distances = dtw.distance_matrix_fast(
            instances,
            block=((0, num_left), (num_left, num_left + num_right)),
        )
        assert distances.shape == (
            num_left,
            num_right,
        ), "Terry please adjust `block` argument if this assert failed."
        return numpy.asarray(distances, dtype=numpy.float32)

    def pairwise(self, instances: numpy.ndarray) -> numpy.ndarray:
        """Compute a 2d array of distances among each pair in `instances`."""
        distances = dtw.distance_matrix_fast(instances)
        return numpy.asarray(distances, dtype=numpy.float32)
