//! Subgraph Cardinality algorithm.

use crate::{
    chaoda::{Graph, Vertex},
    Cluster, DistanceValue,
};

use super::GraphEvaluator;

/// `Cluster`s in subgraphs with relatively small population are more likely to
/// be anomalous.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SubgraphCardinality;

impl<T: DistanceValue, S: Cluster<T>, V: Vertex<T, S>> GraphEvaluator<T, S, V> for SubgraphCardinality {
    fn name(&self) -> &'static str {
        "sc"
    }

    fn evaluate_clusters(&self, g: &Graph<T, S, V>) -> Vec<f32> {
        g.iter_components()
            .flat_map(|sg| {
                let p = -(sg.population() as f32);
                core::iter::repeat_n(p, sg.cardinality())
            })
            .collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}

impl Default for SubgraphCardinality {
    fn default() -> Self {
        Self
    }
}
