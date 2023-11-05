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

// pub mod chaoda;
mod cakes;
mod core;
pub(crate) mod utils;

pub use crate::{
    cakes::{knn, rnn, sharded::ShardedCakes, singular::Cakes},
    core::{
        cluster::{Cluster, PartitionCriteria, PartitionCriterion, SerializedCluster, Tree},
        dataset::{Dataset, Instance, VecDataset},
    },
};

/// The current version of the crate.
pub const VERSION: &str = "0.23.1";
