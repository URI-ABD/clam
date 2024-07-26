//! Stationary Probabilities Algorithm.

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::chaoda::Graph;

use super::Algorithm;

/// Clusters with smaller stationary probabilities are more anomalous.
#[derive(Clone, Serialize, Deserialize)]
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

impl Algorithm for StationaryProbability {
    fn name(&self) -> String {
        format!("sp-{}", self.num_steps)
    }

    fn evaluate_clusters<U: Number>(&self, g: &mut Graph<U>) -> Vec<f32> {
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
