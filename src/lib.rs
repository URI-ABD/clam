#![doc = include_str!("../README.md")]

// pub mod chaoda;
pub mod cakes;
mod core;
pub mod needleman_wunch;
pub(crate) mod utils;

pub use crate::core::{cluster, dataset};

pub const VERSION: &str = "0.17.0";

#[allow(clippy::type_complexity)]
pub const COMMON_METRICS_F32: &[(&str, fn(&[f32], &[f32]) -> f32)] = &[
    ("euclidean", distances::vectors::euclidean),
    ("euclidean_sq", distances::vectors::euclidean_sq),
    ("manhattan", distances::vectors::manhattan),
    ("cosine", distances::vectors::cosine),
];

#[allow(clippy::type_complexity)]
pub const COMMON_METRICS_F64: &[(&str, fn(&[f64], &[f64]) -> f64)] = &[
    ("euclidean", distances::vectors::euclidean),
    ("euclidean_sq", distances::vectors::euclidean_sq),
    ("manhattan", distances::vectors::manhattan),
    ("cosine", distances::vectors::cosine),
];

#[allow(clippy::type_complexity)]
pub const COMMON_METRICS_STR: &[(&str, fn(&str, &str) -> u32)] = &[
    ("hamming", distances::strings::hamming),
    ("levenshtein", distances::strings::levenshtein),
    ("needleman_wunsch", needleman_wunch::nw_distance),
];
