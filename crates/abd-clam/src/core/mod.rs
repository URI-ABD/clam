//! The core traits and structs for CLAM.

pub mod cluster;
pub mod dataset;

pub use cluster::{Ball, Children, Cluster, ClusterAdaptor, ParPartition, Partition, LFD};
pub use dataset::{
    Dataset, FlatVec, LinearSearch, Metric, MetricSpace, ParDataset, ParLinearSearch, ParMetricSpace, Permutable,
    SizedHeap,
};
