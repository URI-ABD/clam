#![doc = include_str!("../README.md")]

pub mod cakes;
mod core;
pub mod pancakes;
pub mod utils;

pub use core::{
    Ball, ClamIO, Cluster, Dataset, DistanceValue, FloatDistanceValue, ParClamIO, ParCluster, ParDataset, ParPartition,
    Partition, SizedHeap, LFD,
};

use core::{MaxItem, MinItem};

#[cfg(feature = "chaoda")]
pub mod chaoda;

#[cfg(feature = "mbed")]
pub mod mbed;

#[cfg(feature = "musals")]
pub mod musals;

/// The current version of the crate.
pub const VERSION: &str = "0.32.0";
