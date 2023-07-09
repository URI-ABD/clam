//! CLAM is a library around learning manifolds in a Banach space defined by a distance metric.
//!
//! # Papers
//!
//! - [CHESS](https://arxiv.org/abs/1908.08551)
//! - [CHAODA](https://arxiv.org/abs/2103.11774)
//!

// pub mod chaoda;
pub mod cakes;
mod core;
pub mod needleman_wunch;
pub mod utils;

pub use crate::core::{cluster, dataset};

pub const VERSION: &str = "0.15.0";
