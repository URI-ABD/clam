//! Subgraph Cardinality algorithm.

use crate::{
    chaoda::{Graph, Vertex},
    DistanceValue,
};

use super::GraphEvaluator;

/// `Cluster`s in subgraphs with relatively small population are more likely to
/// be anomalous.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SubgraphCardinality;

impl<T: DistanceValue, V: Vertex<T>> GraphEvaluator<T, V> for SubgraphCardinality {
    fn evaluate_clusters(&self, g: &Graph<T, V>) -> Vec<f32> {
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
