//! The `Cluster` variants for use in CHAODA.

use std::hash::Hash;

use crate::{Cluster, DistanceValue, ParCluster};

mod odd_ball;

pub use odd_ball::OddBall;

/// The `Vertex` trait is used to represent a node in a graph structure.
///
/// Vertices, and the relationships between them, can then be used to detect
/// anomalies.
pub trait Vertex<T: DistanceValue>: Cluster<T> + Hash {
    /// The number of features in the feature vector.
    const NUM_FEATURES: usize;

    /// The type of the feature vector.
    ///
    /// This is treated as an array of length `NUM_FEATURES`.
    type FeatureVector: AsRef<[f32]>;

    /// Returns the feature vector used for anomaly detection.
    ///
    /// All vertices from the same tree should have the same length feature
    /// vector.
    fn feature_vector(&self) -> Self::FeatureVector;

    /// The accumulated child-parent cardinality ratio.
    fn accumulated_cp_cardinality_ratio(&self) -> f32;
}

/// Parallel version of the `Vertex` trait.
pub trait ParVertex<T: DistanceValue + Send + Sync>: Vertex<T> + ParCluster<T> {}
