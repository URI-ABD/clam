//! A `Node` is more anomalous if it has fewer neighbors in the graph relative to other nodes in the graph.

use crate::{DistanceValue, NamedAlgorithm, Tree, chaoda::Node};

use super::{AnomalyFeatures, Graph, GraphAlg};

/// A `Node` is more anomalous if it has fewer neighbors in the graph relative to other nodes in the graph.
#[derive(Debug, Clone)]
#[must_use]
pub struct RelativeVertexDegree;

impl_named_algorithm_for_unit_struct!(RelativeVertexDegree, "relative-vertex-degree", r"^relative-vertex-degree$");

impl From<RelativeVertexDegree> for super::GraphAlgorithm {
    fn from(alg: RelativeVertexDegree) -> Self {
        Self::RelativeVertexDegree(alg)
    }
}

impl GraphAlg for RelativeVertexDegree {
    fn rank_nodes<'a, Id, I, T, A, M>(&self, graph: &'a Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)>
    where
        T: DistanceValue,
    {
        // The more edges a node has, the less anomalous it is, the higher the rank.
        graph.iter_nodes().map(|n| (n, n.num_edges())).collect()
    }
}
