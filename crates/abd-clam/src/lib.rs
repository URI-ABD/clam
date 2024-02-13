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
    core::{
        cluster::{BaseCluster, Cluster, MaxDepth, MinCardinality, PartitionCriteria, PartitionCriterion},
        dataset::{Dataset, Instance, VecDataset},
        graph::{Edge, Graph, MetaMLScorer, Ratios, Vertex},
        tree::Tree,
    },
};

/// The current version of the crate.
pub const VERSION: &str = "0.28.0";
