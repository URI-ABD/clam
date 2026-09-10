//! A `Node` is more anomalous if it represents a number of items relative to other `Node`s in the `Graph`.

use crate::{DistanceValue, NamedAlgorithm, Tree, chaoda::Node};

use super::{AnomalyFeatures, Graph, GraphAlg};

/// Assign anomaly scores to nodes based on the relative cardinality to other nodes in the graph.
#[derive(Debug, Clone)]
#[must_use]
pub struct RelativeClusterCardinality;

impl_named_algorithm_for_unit_struct!(RelativeClusterCardinality, "relative-cluster-cardinality", r"^relative-cluster-cardinality$");

impl From<RelativeClusterCardinality> for super::GraphAlgorithm {
    fn from(alg: RelativeClusterCardinality) -> Self {
        Self::RelativeClusterCardinality(alg)
    }
}

impl GraphAlg for RelativeClusterCardinality {
    fn rank_nodes<'a, Id, I, T, A, M>(&self, graph: &'a Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)>
    where
        T: DistanceValue,
    {
        // The more items in the node, the less anomalous it is, the higher the rank.
        graph.iter_nodes().map(|n| (n, n.num_items())).collect()
    }
}
