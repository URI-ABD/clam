//! Anomaly detection algorithms using CLAM.

use rayon::prelude::*;

use crate::{DistanceValue, Tree};

use super::{AnomalyFeatures, Graph};

mod accumulated_cardinality_ratios;
mod graph_neighborhood_size;
mod relative_cluster_cardinality;
mod relative_component_cardinality;
mod relative_vertex_degree;
mod stationary_probabilities;

pub use accumulated_cardinality_ratios::AccumulatedCardinalityRatios;
pub use graph_neighborhood_size::GraphNeighborhoodSize;
pub use relative_cluster_cardinality::RelativeClusterCardinality;
pub use relative_component_cardinality::RelativeComponentCardinality;
pub use relative_vertex_degree::RelativeVertexDegree;
pub use stationary_probabilities::StationaryProbabilities;

/// An anomaly detection algorithm that can be applied to a Chaoda graph.
///
/// Implementors of this trait should provide the [`Self::raw_anomaly_scores`] method and users should use the [`Self::anomaly_scores`] method to get normalized
/// anomaly scores in the range [0, 1] with higher scores indicating more anomalous items.
pub trait ChaodaAlgorithm<D, M, T, A>
where
    T: DistanceValue,
{
    /// Compute anomaly scores for all items from the tree used to create the graph.
    ///
    /// High scores indicate more anomalous nodes, and low scores indicate less anomalous nodes. The scores are not normalized, so they can take any value. They
    /// will be normalized by other methods provided with this trait.
    ///
    /// The returned vector should have the same length as the number of items in the tree and the order of the scores should correspond to the order of the
    /// items in the tree.
    fn raw_anomaly_scores(&self, graph: &Graph<T>, tree: &Tree<D, M, T, (A, AnomalyFeatures)>) -> Result<Vec<f64>, String>;

    /// Compute anomaly scores for all items from the tree used to create the graph, normalized to the range [0, 1] using gaussian error function normalization.
    #[expect(clippy::cast_precision_loss)]
    fn anomaly_scores(&self, graph: &Graph<T>, tree: &Tree<D, M, T, (A, AnomalyFeatures)>) -> Result<Vec<f64>, String> {
        let raw_scores = self.raw_anomaly_scores(graph, tree)?;
        let mean_score = raw_scores.iter().copied().sum::<f64>() / raw_scores.len() as f64;
        let std_dev_score = (raw_scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>() / raw_scores.len() as f64).sqrt();
        Ok(raw_scores
            .into_iter()
            // Standardize the scores to have mean 0 and standard deviation 1.
            .map(|s| (s - mean_score) / std_dev_score)
            // Apply the gaussian error function to the standardized scores to the [-1, 1] range.
            .map(libm::erf)
            // Scale the scores to the [0, 1] range.
            .map(|s| f64::midpoint(s, 1.0))
            .collect())
    }
}

/// Parallel extension of the [`ChaodaAlgorithm`] trait.
pub trait ParChaodaAlgorithm<D, M, T, A>: ChaodaAlgorithm<D, M, T, A>
where
    D: Send + Sync,
    M: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
{
    /// Parallel version of [`ChaodaAlgorithm::raw_anomaly_scores`], with the default implementation offering no parallelism.
    fn par_raw_anomaly_scores(&self, graph: &Graph<T>, tree: &Tree<D, M, T, (A, AnomalyFeatures)>) -> Result<Vec<f64>, String> {
        self.raw_anomaly_scores(graph, tree)
    }

    /// Parallel version of [`ChaodaAlgorithm::anomaly_scores`].
    #[expect(clippy::cast_precision_loss)]
    fn par_anomaly_scores(&self, graph: &Graph<T>, tree: &Tree<D, M, T, (A, AnomalyFeatures)>) -> Result<Vec<f64>, String> {
        let raw_scores = self.par_raw_anomaly_scores(graph, tree)?;
        let mean_score = raw_scores.par_iter().copied().sum::<f64>() / raw_scores.len() as f64;
        let std_dev_score = (raw_scores.par_iter().map(|s| (s - mean_score).powi(2)).sum::<f64>() / raw_scores.len() as f64).sqrt();
        Ok(raw_scores
            .into_par_iter()
            // Standardize the scores to have mean 0 and standard deviation 1.
            .map(|s| (s - mean_score) / std_dev_score)
            // Apply the gaussian error function to the standardized scores to the [-1, 1] range.
            .map(libm::erf)
            // Scale the scores to the [0, 1] range.
            .map(|s| f64::midpoint(s, 1.0))
            .collect())
    }
}
