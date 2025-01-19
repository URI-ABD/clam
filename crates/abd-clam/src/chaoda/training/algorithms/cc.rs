//! Cluster Cardinality algorithm.

use distances::Number;

use crate::{chaoda::Graph, Cluster};

use super::GraphEvaluator;

/// `Cluster`s with relatively few points are more likely to be anomalous.
#[derive(Clone)]
#[cfg_attr(feature = "disk-io", derive(serde::Serialize, serde::Deserialize))]
pub struct ClusterCardinality;

impl<T: Number, S: Cluster<T>> GraphEvaluator<T, S> for ClusterCardinality {
    fn name(&self) -> &str {
        "cc"
    }

    fn evaluate_clusters(&self, g: &Graph<T, S>) -> Vec<f32> {
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
