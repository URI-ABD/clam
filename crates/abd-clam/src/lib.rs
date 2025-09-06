#![doc = include_str!("../README.md")]

pub mod cakes;
mod core;
pub mod pancakes;
pub mod utils;

pub use core::{
    cluster::{Adapted, Ball, Cluster, ParCluster, ParPartition, Partition, LFD},
    dataset::{Dataset, ParDataset, Permutable, SizedHeap},
    io::{DiskIO, ParDiskIO},
    DistanceValue, FloatDistanceValue,
};

#[cfg(feature = "chaoda")]
pub mod chaoda;

#[cfg(feature = "mbed")]
pub mod mbed;

#[cfg(feature = "musals")]
pub mod musals;

/// The current version of the crate.
pub const VERSION: &str = "0.32.0";
