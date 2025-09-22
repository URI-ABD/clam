//! Cluster Cardinality algorithm.

use crate::{
    chaoda::{Graph, Vertex},
    DistanceValue,
};

use super::GraphEvaluator;

/// `Cluster`s with relatively few points are more likely to be anomalous.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterCardinality;

impl<T: DistanceValue, V: Vertex<T>> GraphEvaluator<T, V> for ClusterCardinality {
    fn evaluate_clusters(&self, g: &Graph<T, V>) -> Vec<f32> {
        g.iter_vertices().map(|c| -(c.cardinality() as f32)).collect()
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
