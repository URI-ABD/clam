//! CLAM is a library/command-line interface around learning manifolds in a Banach spaced defined by a distance metric.
//!
//! # Papers
//!
//! - [CLAM](https://arxiv.org/abs/1908.08551)
//! - [CHAODA](https://arxiv.org/abs/2103.11774)
//!

// pub mod anomaly;
mod core;
pub mod distances;
pub mod search;
pub mod utils;

pub use crate::core::{cluster, dataset, graph, number};
