//! Subgraph Cardinality algorithm.

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::chaoda::Graph;

use super::Algorithm;

/// `Cluster`s in subgraphs with relatively small population are more likely to be anomalous.
#[derive(Clone, Serialize, Deserialize)]
pub struct SubgraphCardinality;

impl Algorithm for SubgraphCardinality {
    fn name(&self) -> String {
        "sc".to_string()
    }

    fn evaluate_clusters<U: Number>(&self, g: &mut Graph<U>) -> Vec<f32> {
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
