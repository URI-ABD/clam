//! A `Node` is more anomalous if it comes from a cluster whose accumulated cardinality ratio is low.

use crate::{DistanceValue, Tree};

use super::{AnomalyFeatures, ChaodaAlgorithm, Graph, ParChaodaAlgorithm};

/// Assign anomaly scores to nodes based on the accumulated cardinality ratios of their clusters.
#[derive(Debug, Clone)]
#[must_use]
pub struct AccumulatedCardinalityRatios;

impl<D, M, T, A> ChaodaAlgorithm<D, M, T, A> for AccumulatedCardinalityRatios
where
    T: DistanceValue,
{
    fn raw_anomaly_scores(&self, graph: &Graph<T>, tree: &Tree<D, M, T, (A, AnomalyFeatures)>) -> Result<Vec<f64>, String> {
        let clusters = graph
            .iter_nodes()
            .map(|n| tree.get_cluster(n.direct_center_index()).map(|c| (c, c.annotation.1.cardinality_ratio)))
            .collect::<Result<Vec<_>, _>>()?;

        let mut scores = clusters
            .into_iter()
            .flat_map(|(c, score)| c.items_indices().map(move |i| (i, score)))
            .collect::<Vec<_>>();
        scores.sort_by_key(|(i, _)| *i);
        Ok(scores.into_iter().map(|(_, score)| score).collect())
    }
}

impl<D, M, T, A> ParChaodaAlgorithm<D, M, T, A> for AccumulatedCardinalityRatios
where
    D: Send + Sync,
    M: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
{
}
