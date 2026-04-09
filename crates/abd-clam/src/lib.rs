//! Clustering, Learning, and Approximation with Manifolds.
//!
//! We provide functionality for clustering, search, multiple sequence alignment, anomaly detection, compression, compressive search, dimension reduction, and
//! more. All algorithms are designed to work efficiently with large high-dimensional datasets under arbitrary distance functions.
//!
//! ## Algorithm Families and Applications
//!
//! - [`cakes`]: Entropy-Scaling Search. These algorithms leverage the geometric and topological structure inherent in real-world datasets to achieve sub-linear
//!   scaling in search throughput with respect to dataset size, and are designed to work efficiently with large high-dimensional datasets under arbitrary
//!   distance functions.
//! - [`musals`]: Multiple Sequence Alignment. This provides the required functionality to create multiple sequence alignments from a set of sequences, as well
//!   implementations of edit-distance metrics using arbitrary cost matrices. The MSA algorithms use the [`Tree`] as a guide tree, and are designed to scale to
//!   very large collections of sequences.
//! - [`pancakes`]: Compression and compressive search algorithms.
//!
//! ## Features
//!
//! - `serde`: Enables serialization and deserialization of various data structures using [`serde`] and [`databuf`].
//! - `musals`: Enables the [`Tree::align`] and [`Tree::par_align`] methods for creating multiple sequence alignments, as the relevant types and algorithms.
//! - `pancakes`: Enables compression and decompression of [`Tree`]s, the [`CompressiveSearch`](pancakes::CompressiveSearch) trait to support search on
//!   compressed [`Tree`]s, and implementations of this trait for the algorithms in the [`cakes`] module.
//! - `all`: Enables the `serde`, `musals`, and `pancakes` features.
//! - `profile`: Enables some minimal profiling using the [`profi`] crate.
//! - `shell`: Enables the `clam-shell` CLI.

#[macro_use]
pub mod utils;
pub use utils::{DistanceValue, FloatDistanceValue, NamedAlgorithm, common_metrics};

pub mod tree;
pub use tree::{Cluster, PartitionStrategy, Tree};

pub mod cakes;
pub use cakes::Cakes;

#[cfg(feature = "musals")]
pub mod musals;

#[cfg(feature = "pancakes")]
pub mod pancakes;

// #[cfg(feature = "chaoda")]
// pub mod chaoda;

// #[cfg(feature = "mbed")]
// pub mod mbed;
