//! Clustering, Learning, and Approximation with Manifolds.
//!
//! We provide functionality for clustering, search, multiple sequence alignment, anomaly detection, compression, compressive search, dimension reduction, and
//! more. All algorithms are designed to work efficiently with large high-dimensional datasets under arbitrary distance functions.
//!
//! ## Algorithm Families and Applications
//!
//! - [`cakes`]: Search (k-NN, p-NN) algorithms.
//! - [`musals`]: Multiple sequence alignment of genomic and protein sequences.
//! - [`pancakes`]: Compression and compressive search algorithms.
//! - `chaoda`: Anomaly detection algorithms using clustering trees and graphs.
//! - `mbed`: Dimension reduction algorithms.
//!
//! ## Features
//!
//! - `serde`: Enables serialization and deserialization of various data structures using [`serde`] and [`databuf`].
//! - `musals`: Enables the `musals` module for multiple sequence alignment.
//! - `pancakes`: Enables the `pancakes` module for compression and compressive search.
//! - `all`: Enables the `serde`, `musals`, and `pancakes` features.
//! - `profile`: Enables profiling using the [`profi`] crate.

mod tree;
mod utils;

pub use tree::{Cluster, PartitionStrategy, Tree, partition_strategy};
pub use utils::DistanceValue;

pub mod cakes;

#[cfg(feature = "musals")]
pub mod musals;

#[cfg(feature = "pancakes")]
pub mod pancakes;

// #[cfg(feature = "mbed")]
// pub mod mbed;
