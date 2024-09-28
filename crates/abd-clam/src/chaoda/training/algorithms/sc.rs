//! Subgraph Cardinality algorithm.

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::{chaoda::Graph, Cluster, Dataset};

use super::GraphEvaluator;

/// `Cluster`s in subgraphs with relatively small population are more likely to be anomalous.
#[derive(Clone, Serialize, Deserialize)]
pub struct SubgraphCardinality;

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> GraphEvaluator<I, U, D, S> for SubgraphCardinality {
    fn name(&self) -> &str {
        "sc"
    }

    fn evaluate_clusters(&self, g: &Graph<I, U, D, S>) -> Vec<f32> {
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
