//! Vertex Degree Algorithm

use crate::{
    chaoda::{Graph, Vertex},
    Cluster, DistanceValue,
};

use super::GraphEvaluator;

/// `Cluster`s with relatively few neighbors are more likely to be anomalous.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct VertexDegree;

impl<T: DistanceValue, S: Cluster<T>, V: Vertex<T, S>> GraphEvaluator<T, S, V> for VertexDegree {
    fn name(&self) -> &'static str {
        "vd"
    }

    fn evaluate_clusters(&self, g: &Graph<T, S, V>) -> Vec<f32> {
        g.iter_neighbors().map(|n| -(n.len() as f32)).collect()
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
