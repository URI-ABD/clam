//! The core traits and structs for CLAM.

pub mod adapters;
pub mod cluster;
pub mod dataset;
pub mod metric;
pub mod tree;

pub use cluster::{Ball, Cluster, LFD};
pub use dataset::{Dataset, FlatVec, SizedHeap};
pub use metric::Metric;

#[cfg(feature = "disk-io")]
mod io;

#[cfg(feature = "disk-io")]
pub use io::{DiskIO, ParDiskIO};
