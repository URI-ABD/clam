//! Clustering, Learning, and Approximation with Manifolds.
//!
//! We provide functionality for clustering, search, multiple sequence alignment, anomaly detection, compression, compressive search, dimension reduction, and
//! more. All algorithms are designed to work efficiently with large high-dimensional datasets under arbitrary distance functions.
//!
//! ## Algorithm Families and Applications
//!
//! - [`cakes`]: Search (k-NN, p-NN) algorithms.
//! - [`musals`]: Multiple sequence alignment of genomic and protein sequences. Use the `musals` feature to enable this module.
//! - `chaoda`: Anomaly detection algorithms using clustering trees and graphs. Use the `chaoda` feature to enable this module. WIP.
//! - `codec`: Compression and compressive search algorithms. Use the `codec` feature to enable this module. WIP.
//! - `mbed`: Dimension reduction algorithms. Use the `mbed` feature to enable this module. WIP.
//!
//! ## Features
//!
//! - `serde`: Enables serialization and deserialization of clustering trees and related data structures using the [`serde`] and [`databuf`] crates.
//! - `musals`: Enables the `musals` module for multiple sequence alignment.
//! - `all`: Enables the `serde` and `musals` features.
//! - `profile`: Enables profiling using the [`profi`] crate.

pub mod cakes;
mod tree;
mod utils;

pub use tree::{Cluster, PartitionStrategy, Tree, partition_strategy};

pub use utils::{DistanceValue, FloatDistanceValue};

#[cfg(feature = "musals")]
pub mod musals;

// #[cfg(feature = "codec")]
// pub mod codec;

// #[cfg(feature = "mbed")]
// pub mod mbed;
