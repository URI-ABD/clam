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
pub mod needleman_wunch;
pub(crate) mod utils;

pub(crate) use crate::core::cluster::Cluster;
pub use crate::{
    cakes::{knn, rnn, Cakes},
    core::{
        cluster::{PartitionCriteria, PartitionCriterion, Tree},
        dataset::{Dataset, VecDataset},
    },
};

/// The current version of the crate.
pub const VERSION: &str = "0.20.4-dev0";

/// Common distance functions and their names for slices of `f32`.
#[allow(clippy::type_complexity)]
pub const COMMON_METRICS_F32: &[(&str, fn(&[f32], &[f32]) -> f32)] = &[
    ("euclidean", distances::vectors::euclidean),
    ("euclidean_sq", distances::vectors::euclidean_sq),
    ("manhattan", distances::vectors::manhattan),
    ("cosine", distances::vectors::cosine),
];

/// Common distance functions and their names for slices of `f64`.
#[allow(clippy::type_complexity)]
pub const COMMON_METRICS_F64: &[(&str, fn(&[f64], &[f64]) -> f64)] = &[
    ("euclidean", distances::vectors::euclidean),
    ("euclidean_sq", distances::vectors::euclidean_sq),
    ("manhattan", distances::vectors::manhattan),
    ("cosine", distances::vectors::cosine),
];

/// Common distance functions and their names for `&str`.
#[allow(clippy::type_complexity)]
pub const COMMON_METRICS_STR: &[(&str, fn(&str, &str) -> u32)] = &[
    ("hamming", distances::strings::hamming),
    ("levenshtein", distances::strings::levenshtein),
    ("needleman_wunsch", needleman_wunch::nw_distance),
];
