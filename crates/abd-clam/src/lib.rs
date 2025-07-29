#![doc = include_str!("../README.md")]

pub mod cakes;
mod core;
pub mod pancakes;
pub mod utils;

pub use core::{adapters, cluster, dataset, metric, tree, Ball, Cluster, Dataset, FlatVec, Metric, SizedHeap, LFD};

#[cfg(feature = "disk-io")]
pub use core::{DiskIO, ParDiskIO};

#[cfg(feature = "chaoda")]
pub mod chaoda;

#[cfg(feature = "mbed")]
pub mod mbed;

#[cfg(feature = "musals")]
pub mod musals;

/// The current version of the crate.
pub const VERSION: &str = "0.32.0";
