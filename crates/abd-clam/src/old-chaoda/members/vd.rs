//! Vertex Degree Algorithm

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::chaoda::Graph;

use super::Algorithm;

/// `Cluster`s with relatively few neighbors are more likely to be anomalous.
#[derive(Clone, Serialize, Deserialize)]
pub struct VertexDegree;

impl Algorithm for VertexDegree {
    fn name(&self) -> String {
        "vd".to_string()
    }

    fn evaluate_clusters<U: Number>(&self, g: &mut Graph<U>) -> Vec<f32> {
        g.iter_neighbors().map(|n| -n.len().as_f32()).collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}

impl Default for VertexDegree {
    fn default() -> Self {
        Self
    }
}
