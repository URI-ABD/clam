"""Pydantic models for Criterion benchmarks."""

import json
import pathlib
import typing

import numpy
import pydantic


class Benchmark(pydantic.BaseModel):
    """Summary of a Criterion benchmark."""

    group_id: str
    function_id: str
    value_str: str
    throughput: dict[str, int]
    full_id: str
    directory_name: str
    title: str

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "Benchmark":
        """Load a benchmark from a path."""
        with path.open("r") as file:
            data = json.load(file)

        return cls(**data)

    @property
    def num_queries(self) -> int:
        """Return the number of queries."""
        return self.throughput["Elements"]


class ConfidenceInterval(pydantic.BaseModel):
    """Confidence interval for a measurement."""

    confidence_level: float
    lower_bound: float
    upper_bound: float

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize a ConfidenceInterval."""
        super().__init__(**kwargs)

        # Convert nanoseconds to seconds
        self.lower_bound *= 1e-9
        self.upper_bound *= 1e-9

    @property
    def mean(self) -> float:
        """Return the mean of the elapsed times."""
        return (self.lower_bound + self.upper_bound) / 2

    @property
    def std_dev(self) -> float:
        """Return the standard deviation of the elapsed times."""
        return (self.upper_bound - self.lower_bound) / (2 * 1.96)


class Measurement(pydantic.BaseModel):
    """Summary of a Criterion measurement."""

    confidence_interval: ConfidenceInterval
    point_estimate: float
    standard_error: float

    @property
    def mean(self) -> float:
        """Return the mean of the elapsed times."""
        return self.confidence_interval.mean

    @property
    def std_dev(self) -> float:
        """Return the standard deviation of the elapsed times."""
        return self.confidence_interval.std_dev


class Estimates(pydantic.BaseModel):
    """Summary of a Criterion benchmark."""

    mean: Measurement
    median: Measurement
    median_abs_dev: Measurement
    slope: typing.Optional[Measurement]
    std_dev: Measurement

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "Estimates":
        """Load an estimate from a path."""
        with path.open("r") as file:
            data = json.load(file)

        return cls(**data)

    @property
    def elapsed_mean(self) -> float:
        """Return the mean of the elapsed times."""
        return self.mean.mean

    @property
    def elapsed_std(self) -> float:
        """Return the standard deviation of the elapsed times."""
        return self.std_dev.mean


class Sample(pydantic.BaseModel):
    """Criterion Sampling."""

    sampling_mode: str
    iters: list[float]
    times: list[float]

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "Sample":
        """Load a sample from a path."""
        with path.open("r") as file:
            data = json.load(file)

        return cls(**data)


class Ks(pydantic.BaseModel):
    """Benchmark results for a specific k."""

    k: int
    benchmark: Benchmark
    estimates: Estimates
    sample: Sample

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "Ks":
        """Load a base from a path."""
        k = int(path.name)

        with path.joinpath("base", "benchmark.json").open("r") as file:
            benchmark = Benchmark(**json.load(file))

        with path.joinpath("base", "estimates.json").open("r") as file:
            estimates = Estimates(**json.load(file))

        with path.joinpath("base", "sample.json").open("r") as file:
            sample = Sample(**json.load(file))

        return cls(k=k, benchmark=benchmark, estimates=estimates, sample=sample)

    @property
    def num_queries(self) -> int:
        """Return the number of queries."""
        return self.benchmark.num_queries

    @property
    def elapsed_mean(self) -> float:
        """Return the mean of the elapsed times."""
        return self.estimates.elapsed_mean / self.num_queries

    @property
    def elapsed_std(self) -> float:
        """Return the standard deviation of the elapsed times."""
        return self.estimates.elapsed_std / self.num_queries

    @property
    def elapsed(self) -> list[float]:
        """Return a randomly sampled list of elapsed times."""
        rng = numpy.random.default_rng()
        samples = rng.normal(self.elapsed_mean, self.elapsed_std, self.num_queries)
        return list(map(float, samples))


class ShardAlgorithm(pydantic.BaseModel):
    """Benchmark for a combination of num-shards and algorithm."""

    num_shards: int
    algorithm: str
    ks: list[Ks]

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "ShardAlgorithm":
        """Load a base from a path."""
        name = path.name.split("-")

        num_shards = int(name[1])
        algorithm = name[2]
        ks = [
            Ks.from_path(p) for p in path.iterdir() if p.is_dir() and p.name != "report"
        ]

        return cls(num_shards=num_shards, algorithm=algorithm, ks=ks)


class ShardGroup(pydantic.BaseModel):
    """Benchmark results for a specific dataset."""

    data_name: str
    benches: dict[int, list[ShardAlgorithm]]

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "ShardGroup":
        """Load a Group from a path."""
        data_name = path.name[8:]
        shard_algorithms = [
            ShardAlgorithm.from_path(p)
            for p in path.iterdir()
            if p.name.startswith("knn-") or p.name.startswith("rnn-")
        ]

        benches: dict[int, list[ShardAlgorithm]] = {}
        for sa in shard_algorithms:
            if sa.num_shards not in benches:
                benches[sa.num_shards] = []
            benches[sa.num_shards].append(sa)

        return cls(data_name=data_name, benches=benches)
