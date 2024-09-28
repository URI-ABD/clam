//! Cluster Cardinality algorithm.

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::{chaoda::Graph, Cluster, Dataset};

use super::GraphEvaluator;

/// `Cluster`s with relatively few points are more likely to be anomalous.
#[derive(Clone, Serialize, Deserialize)]
pub struct ClusterCardinality;

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> GraphEvaluator<I, U, D, S> for ClusterCardinality {
    fn name(&self) -> &str {
        "cc"
    }

    fn evaluate_clusters(&self, g: &Graph<I, U, D, S>) -> Vec<f32> {
        g.iter_clusters().map(|c| -c.cardinality().as_f32()).collect()
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
