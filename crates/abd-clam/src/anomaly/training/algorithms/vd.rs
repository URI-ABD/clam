//! Vertex Degree Algorithm

use crate::{
    chaoda::{Graph, Vertex},
    DistanceValue,
};

use super::GraphEvaluator;

/// `Cluster`s with relatively few neighbors are more likely to be anomalous.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct VertexDegree;

impl<T: DistanceValue, V: Vertex<T>> GraphEvaluator<T, V> for VertexDegree {
    fn evaluate_clusters(&self, g: &Graph<T, V>) -> Vec<f32> {
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
