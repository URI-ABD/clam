//! CLAM is a library/command-line interface around learning manifolds in a Banach spaced defined by a distance metric.
//!
//! # Papers
//!
//! - [CLAM](https://arxiv.org/abs/1908.08551)
//! - [CHAODA](https://arxiv.org/abs/2103.11774)
//!

mod anomaly;
mod core;
mod search;
mod traits;

pub mod prelude;
pub mod utils;

pub use crate::anomaly::Chaoda;

pub use crate::core::criteria;
pub use crate::core::Cluster;
pub use crate::core::Edge;
pub use crate::core::Graph;
pub use crate::core::Manifold;

pub use crate::search::codec;
pub use crate::search::Cakes;
pub use crate::search::CompressibleDataset;

pub use crate::traits::dataset;
pub use crate::traits::metric;
pub use crate::traits::Dataset;
pub use crate::traits::Metric;
pub use crate::traits::Number;

pub use crate::anomaly::get_individual_algorithms;
pub use crate::anomaly::get_meta_ml_methods;
pub use crate::traits::metric::metric_from_name;
