//! A `Node` is more anomalous if it is visited less frequently during an infinite random walk on the graph.

use rayon::prelude::*;

use crate::{DistanceValue, Tree};

use super::{AnomalyFeatures, ChaodaAlgorithm, Graph, ParChaodaAlgorithm};

/// A `Node` is more anomalous if it is visited less frequently during an infinite random walk on the graph.
#[derive(Debug, Clone)]
#[must_use]
pub struct StationaryProbabilities;

impl<D, M, T, A> ChaodaAlgorithm<D, M, T, A> for StationaryProbabilities
where
    T: DistanceValue,
{
    fn raw_anomaly_scores(&self, graph: &Graph<T>, _: &Tree<D, M, T, (A, AnomalyFeatures)>) -> Result<Vec<f64>, String> {
        let mut scores = graph
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

                // Join the nodes with their stationary probabilities and convert to anomaly scores.
                nodes.into_iter().zip(row_sums).flat_map(|(node, stationary_prob)| {
                    // A node is less anomalous if it has a higher stationary probability, thus the negative sign.
                    node.iter_items().map(move |i| (i, -stationary_prob))
                })
            })
            .collect::<Vec<_>>();
        scores.sort_by_key(|(i, _)| *i);
        Ok(scores.into_iter().map(|(_, score)| score).collect())
    }
}

impl<D, M, T, A> ParChaodaAlgorithm<D, M, T, A> for StationaryProbabilities
where
    D: Send + Sync,
    M: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
{
    fn par_raw_anomaly_scores(&self, graph: &Graph<T>, _: &Tree<D, M, T, (A, AnomalyFeatures)>) -> Result<Vec<f64>, String> {
        let mut scores = graph
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

                // Join the nodes with their stationary probabilities and convert to anomaly scores.
                nodes.into_par_iter().zip(row_sums).flat_map(|(node, stationary_prob)| {
                    // A node is less anomalous if it has a higher stationary probability, thus the negative sign.
                    node.par_iter_items().map(move |i| (i, -stationary_prob))
                })
            })
            .collect::<Vec<_>>();

        scores.sort_by_key(|(i, _)| *i);
        Ok(scores.into_iter().map(|(_, score)| score).collect())
    }
}
