//! Relative Parent Cardinality algorithm.

use distances::Number;

use crate::{chaoda::Graph, Cluster};

use super::GraphEvaluator;

/// `Cluster`s with a smaller fraction of points from their parent `Cluster` are
/// more anomalous.
#[derive(Clone)]
#[cfg_attr(feature = "disk-io", derive(serde::Serialize, serde::Deserialize))]
pub struct ParentCardinality;

impl<T: Number, S: Cluster<T>> GraphEvaluator<T, S> for ParentCardinality {
    fn name(&self) -> &str {
        "pc"
    }

    fn evaluate_clusters(&self, g: &Graph<T, S>) -> Vec<f32> {
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
