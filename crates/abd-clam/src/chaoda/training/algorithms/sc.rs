//! Subgraph Cardinality algorithm.

use distances::Number;

use crate::{chaoda::Graph, Cluster};

use super::GraphEvaluator;

/// `Cluster`s in subgraphs with relatively small population are more likely to
/// be anomalous.
#[derive(Clone)]
#[cfg_attr(feature = "disk-io", derive(serde::Serialize, serde::Deserialize))]
pub struct SubgraphCardinality;

impl<T: Number, S: Cluster<T>> GraphEvaluator<T, S> for SubgraphCardinality {
    fn name(&self) -> &str {
        "sc"
    }

    fn evaluate_clusters(&self, g: &Graph<T, S>) -> Vec<f32> {
        g.iter_components()
            .flat_map(|sg| {
                let p = -sg.population().as_f32();
                core::iter::repeat(p).take(sg.cardinality())
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
