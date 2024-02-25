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

mod cakes;
pub mod chaoda;
pub mod codec;
mod core;
pub mod utils;

pub use crate::{
    cakes::{knn, rnn, Cakes},
    chaoda::graph,
    core::{
        cluster::{Cluster, MaxDepth, MinCardinality, PartitionCriteria, PartitionCriterion, UniBall},
        dataset::{Dataset, Instance, VecDataset},
        tree::Tree,
    },
};

/// The current version of the crate.
pub const VERSION: &str = "0.29.0";
