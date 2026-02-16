//! Stationary Probabilities Algorithm.

use crate::{
    chaoda::{Graph, Vertex},
    DistanceValue,
};

use super::GraphEvaluator;

/// Clusters with smaller stationary probabilities are more anomalous.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct StationaryProbability {
    /// The Random Walk will be simulated for 2^`num_steps` steps.
    num_steps: usize,
}

impl StationaryProbability {
    /// Create a new `StationaryProbability` algorithm.
    ///
    /// # Arguments
    ///
    /// * `num_steps`: The Random Walk will be simulated for 2^`num_steps` steps.
    pub const fn new(num_steps: usize) -> Self {
        Self { num_steps }
    }
}

impl<T: DistanceValue, V: Vertex<T>> GraphEvaluator<T, V> for StationaryProbability {
    fn evaluate_clusters(&self, g: &Graph<T, V>) -> Vec<f32> {
        g.compute_stationary_probabilities(self.num_steps)
            .into_iter()
            .map(|x| 1.0 - x)
            .collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}

impl Default for StationaryProbability {
    fn default() -> Self {
        Self { num_steps: 16 }
    }
}
