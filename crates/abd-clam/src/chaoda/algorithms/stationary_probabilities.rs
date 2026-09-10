//! A `Node` is more anomalous if it is visited less frequently during an infinite random walk on the graph.

use rayon::prelude::*;

use crate::{DistanceValue, NamedAlgorithm, Tree, chaoda::Node};

use super::{AnomalyFeatures, Graph, GraphAlg};

/// A `Node` is more anomalous if it is visited less frequently during an infinite random walk on the graph.
#[derive(Debug, Clone)]
#[must_use]
pub struct StationaryProbabilities;

impl_named_algorithm_for_unit_struct!(StationaryProbabilities, "stationary-probabilities", r"^stationary-probabilities$");

impl From<StationaryProbabilities> for super::GraphAlgorithm {
    fn from(alg: StationaryProbabilities) -> Self {
        Self::StationaryProbabilities(alg)
    }
}

impl GraphAlg for StationaryProbabilities {
    fn rank_nodes<'a, Id, I, T, A, M>(&self, graph: &'a Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)>
    where
        T: DistanceValue,
    {
        graph
            .iter_components()
            .flat_map(|c| {
                let (mut matrix, nodes) = c.transition_probability_matrix();

                // Repeatedly square the matrix to approximate the stationary distribution. Squaring the matrix 20 times represents 2^20 steps, which is more
                // than enough for convergence in practice.
                for _ in 0..20 {
                    matrix = matrix.dot(&matrix);
                }

                // Sum up the rows of the matrix to get the stationary probability for each node.
                let row_sums = matrix.outer_iter().map(|row| row.sum()).collect::<Vec<_>>();

                // Join the nodes with their stationary probabilities and sort by stationary probability.
                let mut scores = nodes.into_iter().zip(row_sums).collect::<Vec<_>>();
                scores.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                // Convert the sorted scores into ranks.
                scores.into_iter().enumerate().map(|(rank, (node, _))| (node, rank)).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn par_rank_nodes<'a, Id, I, T, A, M>(&self, graph: &'a Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        graph
            .par_iter_components()
            .flat_map(|c| {
                let (mut matrix, nodes) = c.transition_probability_matrix();

                // Repeatedly square the matrix to approximate the stationary distribution. Squaring the matrix 20 times represents 2^20 steps, which is more
                // than enough for convergence in practice.
                for _ in 0..20 {
                    matrix = matrix.dot(&matrix);
                }

                // Sum up the rows of the matrix to get the stationary probability for each node.
                let row_sums = matrix.outer_iter().map(|row| row.sum()).collect::<Vec<_>>();

                // Join the nodes with their stationary probabilities and sort by stationary probability.
                let mut scores = nodes.into_iter().zip(row_sums).collect::<Vec<_>>();
                scores.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                // Convert the sorted scores into ranks.
                scores.into_iter().enumerate().map(|(rank, (node, _))| (node, rank)).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }
}
