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
pub const VERSION: &str = "0.21.2";

/// Common distance functions and their names.
#[allow(clippy::type_complexity)]
pub const COMMON_METRICS_F32: &[(&str, fn(&Vec<f32>, &Vec<f32>) -> f32)] = &[
    ("euclidean", euclidean_f32),
    ("euclidean_sq", euclidean_sq_f32),
    ("manhattan", manhattan_f32),
    ("cosine", cosine_f32),
];

/// Euclidean distance.
#[allow(clippy::ptr_arg)]
fn euclidean_f32(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    distances::vectors::euclidean(a, b)
}

/// Squared euclidean distance.
#[allow(clippy::ptr_arg)]
fn euclidean_sq_f32(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    distances::vectors::euclidean_sq(a, b)
}

/// Manhattan distance.
#[allow(clippy::ptr_arg)]
fn manhattan_f32(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    distances::vectors::manhattan(a, b)
}

/// Cosine distance.
#[allow(clippy::ptr_arg)]
fn cosine_f32(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    distances::vectors::cosine(a, b)
}

/// Common distance functions and their names.
#[allow(clippy::type_complexity)]
pub const COMMON_METRICS_F64: &[(&str, fn(&Vec<f64>, &Vec<f64>) -> f64)] = &[
    ("euclidean", euclidean_f64),
    ("euclidean_sq", euclidean_sq_f64),
    ("manhattan", manhattan_f64),
    ("cosine", cosine_f64),
];

/// Euclidean distance.
#[allow(clippy::ptr_arg)]
fn euclidean_f64(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    distances::vectors::euclidean(a, b)
}

/// Euclidean squared distance.
#[allow(clippy::ptr_arg)]
fn euclidean_sq_f64(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    distances::vectors::euclidean_sq(a, b)
}

/// Manhattan distance.
#[allow(clippy::ptr_arg)]
fn manhattan_f64(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    distances::vectors::manhattan(a, b)
}

/// Cosine distance.
#[allow(clippy::ptr_arg)]
fn cosine_f64(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    distances::vectors::cosine(a, b)
}

/// Common distance functions and their names.
#[allow(clippy::type_complexity)]
pub const COMMON_METRICS_STR: &[(&str, fn(&String, &String) -> u32)] = &[
    ("hamming", hamming),
    ("levenshtein", levenshtein),
    ("needleman_wunsch", nw_distance),
];

/// Hamming distance.
#[allow(clippy::ptr_arg)]
fn hamming(a: &String, b: &String) -> u32 {
    distances::strings::hamming(a, b)
}

/// Levenshtein distance.
#[allow(clippy::ptr_arg)]
fn levenshtein(a: &String, b: &String) -> u32 {
    distances::strings::levenshtein(a, b)
}

/// Needleman-Wunsch distance.
#[allow(clippy::ptr_arg)]
fn nw_distance(a: &String, b: &String) -> u32 {
    needleman_wunch::nw_distance(a, b)
}
