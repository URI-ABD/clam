//! Relative Parent Cardinality algorithm.

use crate::{
    chaoda::{Graph, Vertex},
    Cluster, DistanceValue,
};

use super::GraphEvaluator;

/// `Cluster`s with a smaller fraction of points from their parent `Cluster` are
/// more anomalous.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ParentCardinality;

impl<T: DistanceValue, S: Cluster<T>, V: Vertex<T, S>> GraphEvaluator<T, S, V> for ParentCardinality {
    fn name(&self) -> &'static str {
        "pc"
    }

    fn evaluate_clusters(&self, g: &Graph<T, S, V>) -> Vec<f32> {
        g.iter_accumulated_cp_car_ratios().collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}

impl Default for ParentCardinality {
    fn default() -> Self {
        Self
    }
}
