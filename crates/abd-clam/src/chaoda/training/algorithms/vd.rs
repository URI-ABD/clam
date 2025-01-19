//! Vertex Degree Algorithm

use distances::Number;

use crate::{chaoda::Graph, Cluster};

use super::GraphEvaluator;

/// `Cluster`s with relatively few neighbors are more likely to be anomalous.
#[derive(Clone)]
#[cfg_attr(feature = "disk-io", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexDegree;

impl<T: Number, S: Cluster<T>> GraphEvaluator<T, S> for VertexDegree {
    fn name(&self) -> &str {
        "vd"
    }

    fn evaluate_clusters(&self, g: &Graph<T, S>) -> Vec<f32> {
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
