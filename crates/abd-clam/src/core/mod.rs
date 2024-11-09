//! The core traits and structs for CLAM.

pub mod cluster;
pub mod dataset;

pub use cluster::{adapter, partition, Ball, Cluster, Partition, LFD};
pub use dataset::{Dataset, FlatVec, Metric, MetricSpace, Permutable};
