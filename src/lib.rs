//! CLAM is a library/command-line interface around learning manifolds in a Banach spaced defined by a distance metric.
//!
//! # Papers
//!
//! - [CLAM](https://arxiv.org/abs/1908.08551)
//! - [CHAODA](https://arxiv.org/abs/2103.11774)
//!

// pub mod anomaly;
// pub mod classification;
pub mod core;
pub mod geometry;
pub mod prelude;
pub mod search;
pub mod traits;
pub mod utils;

// pub use crate::anomaly::CHAODA;
// pub use crate::anomaly_detection::get_individual_algorithms;
// pub use crate::anomaly_detection::get_meta_ml_methods;

pub use crate::core::partition_criteria::PartitionCriteria;
pub use crate::core::partition_criteria::PartitionCriterion;
// pub use crate::core::graph_criteria;
pub use crate::core::Cluster;
// pub use crate::core::Edge;
// pub use crate::core::Graph;
pub use crate::core::Ratios;

pub use crate::search::CAKES;

pub use crate::traits::dataset;
pub use crate::traits::dataset::Tabular;
pub use crate::traits::metric;
pub use crate::traits::metric::metric_from_name;
pub use crate::traits::space::TabularSpace;
pub use crate::traits::Dataset;
pub use crate::traits::Metric;
pub use crate::traits::Number;
pub use crate::traits::Space;
