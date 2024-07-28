#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::style,
    clippy::complexity,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
#![doc = include_str!("../README.md")]

pub mod cakes;
// pub mod chaoda;
mod core;
// pub mod pancakes;
pub mod utils;

pub use crate::{
    cakes::Shardable,
    core::{
        Adapter, Ball, Children, Cluster, Dataset, FlatVec, LinearSearch, Metric, MetricSpace, ParDataset,
        ParLinearSearch, ParMetricSpace, ParPartition, Params, Partition, Permutable, SizedHeap, LFD,
    },
};

/// The current version of the crate.
pub const VERSION: &str = "0.31.0";
