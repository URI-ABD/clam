//! Cluster Cardinality algorithm.

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::chaoda::Graph;

use super::Algorithm;

/// `Cluster`s with relatively few points are more likely to be anomalous.
#[derive(Clone, Serialize, Deserialize)]
pub struct ClusterCardinality;

impl Algorithm for ClusterCardinality {
    fn name(&self) -> String {
        "cc".to_string()
    }

    fn evaluate_clusters<U: Number>(&self, g: &mut Graph<U>) -> Vec<f32> {
        g.iter_clusters().map(|&(_, c)| -c.as_f32()).collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}

impl Default for ClusterCardinality {
    fn default() -> Self {
        Self
    }
}
