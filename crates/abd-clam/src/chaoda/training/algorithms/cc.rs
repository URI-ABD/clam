//! Cluster Cardinality algorithm.

use crate::{
    chaoda::{Graph, Vertex},
    Cluster, DistanceValue,
};

use super::GraphEvaluator;

/// `Cluster`s with relatively few points are more likely to be anomalous.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterCardinality;

impl<T: DistanceValue, S: Cluster<T>, V: Vertex<T, S>> GraphEvaluator<T, S, V> for ClusterCardinality {
    fn name(&self) -> &'static str {
        "cc"
    }

    fn evaluate_clusters(&self, g: &Graph<T, S, V>) -> Vec<f32> {
        g.iter_clusters().map(|c| -(c.cardinality() as f32)).collect()
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
