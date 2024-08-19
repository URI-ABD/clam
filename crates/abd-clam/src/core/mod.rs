//! The core traits and structs for CLAM.

pub mod cluster;
pub mod dataset;

pub use cluster::{adapter, partition, BalancedBall, Ball, Cluster, Partition, LFD};
pub use dataset::{linear_search, Dataset, FlatVec, Metric, MetricSpace, Permutable};
