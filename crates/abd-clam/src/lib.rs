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
mod core;
pub mod pancakes;
pub mod utils;

pub use core::{cluster, dataset, metric, Ball, Cluster, Dataset, FlatVec, Metric, SizedHeap, Tree, LFD};

#[cfg(feature = "chaoda")]
pub mod chaoda;

#[cfg(feature = "mbed")]
pub mod mbed;

#[cfg(feature = "msa")]
pub mod msa;

/// The current version of the crate.
pub const VERSION: &str = "0.32.0";
