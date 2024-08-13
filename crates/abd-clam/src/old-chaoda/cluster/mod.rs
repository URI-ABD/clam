//! Variants of `Cluster` that are used for anomaly detection.

mod vertex;

use distances::Number;

use crate::Cluster;

pub use vertex::{Ratios, Vertex};

/// A cluster that is used for anomaly detection.
pub trait OddBall<U: Number>: Cluster<U> {
    /// Return the properties of the `Cluster` that are used for anomaly detection.
    fn ratios(&self) -> Vec<f32>;

    /// Return the accumulated child-parent cardinality ratio.
    fn accumulated_cp_car_ratio(&self) -> f32;
}
