//! Relative Parent Cardinality algorithm.

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::{chaoda::Graph, Cluster, Dataset};

use super::GraphEvaluator;

/// `Cluster`s with a smaller fraction of points from their parent `Cluster` are more anomalous.
#[derive(Clone, Serialize, Deserialize)]
pub struct ParentCardinality;

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> GraphEvaluator<I, U, D, S> for ParentCardinality {
    fn name(&self) -> &str {
        "pc"
    }

    fn evaluate_clusters(&self, g: &Graph<I, U, D, S>) -> Vec<f32> {
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
