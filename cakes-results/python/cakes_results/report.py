"""Pydantic models for benchmarks and reports from Rust."""

import json
import pathlib

import pydantic


class Report(pydantic.BaseModel):
    """Report from Rust."""

    data_name: str
    metric_name: str
    cardinality: int
    dimensionality: int
    shard_sizes: tuple[int, ...]
    num_queries: int
    k: int
    algorithm: str
    throughput: float
    recall: float
    linear_throughput: float

    def __str__(self) -> str:
        """A summary of the report."""
        return (
            f"Report(\n"
            f"  data_name={self.data_name},\n"
            f"  metric_name={self.metric_name},\n"
            f"  cardinality={self.cardinality},\n"
            f"  dimensionality={self.dimensionality},\n"
            f"  num_queries={self.num_queries},\n"
            f"  k={self.k},\n"
            f"  algorithm={self.algorithm},\n"
            f"  throughput={self.throughput:.3e} QPS,\n"
            f"  linear_throughput={self.linear_throughput:.3e} QPS,\n"
            f"  recall={self.recall:.3e},\n"
            f")"
        )

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "Report":
        """Load a report from a path."""
        with path.open("r") as f:
            return cls(**json.load(f))

    @property
    def num_shards(self) -> int:
        """The number of shards."""
        return len(self.shard_sizes) + 1
