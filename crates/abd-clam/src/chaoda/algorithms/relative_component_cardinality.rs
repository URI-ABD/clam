//! A `Node` is more anomalous if it is in a `Component` whose nodes collectively have fewer items than the other `Components` in the graph.

use crate::{DistanceValue, NamedAlgorithm, Tree};

use super::{super::Node, AnomalyFeatures, Graph, GraphAlg};

/// A `Node` is more anomalous if it is in a `Component` whose nodes collectively have fewer items than the other `Components` in the graph.
#[derive(Debug, Clone)]
#[must_use]
pub struct RelativeComponentCardinality;

impl_named_algorithm_for_unit_struct!(
    RelativeComponentCardinality,
    "relative-component-cardinality",
    r"^relative-component-cardinality$"
);

impl From<RelativeComponentCardinality> for super::GraphAlgorithm {
    fn from(alg: RelativeComponentCardinality) -> Self {
        Self::RelativeComponentCardinality(alg)
    }
}

impl GraphAlg for RelativeComponentCardinality {
    fn rank_nodes<'a, Id, I, T, A, M>(&self, graph: &'a Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)>
    where
        T: DistanceValue,
    {
        graph
            .iter_components()
            .flat_map(|c| {
                let score = c.iter_nodes().map(Node::num_items).sum::<usize>();
                c.iter_nodes().map(move |n| (n, score))
            })
            .collect::<Vec<_>>()
    }
}
