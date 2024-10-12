//! Vertex Degree Algorithm

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::{chaoda::Graph, Cluster, Dataset};

use super::GraphEvaluator;

/// `Cluster`s with relatively few neighbors are more likely to be anomalous.
#[derive(Clone, Serialize, Deserialize)]
pub struct VertexDegree;

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> GraphEvaluator<I, U, D, S> for VertexDegree {
    fn name(&self) -> &str {
        "vd"
    }

    fn evaluate_clusters(&self, g: &Graph<I, U, D, S>) -> Vec<f32> {
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
