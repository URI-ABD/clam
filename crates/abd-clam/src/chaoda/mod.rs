//! Anomaly detection algorithms using CLAM.
//!
//! This module contains CLAM-CHAODA (Clustered Hierarchical Anomaly and Outlier Detection Algorithms). This is a family of algorithms that use CLAM trees to
//! impute graphs that enable unsupervised anomaly detection algorithms.
//!
//! For trees, this enables the [`Tree::annotate_anomaly_features`](crate::Tree::annotate_anomaly_features) method (along with its parallel version). These
//! features can then be used for creating CHAODA graphs. The graphs can, in turn, be used for anomaly detection using the algorithms we provide...

use crate::{DistanceValue, Tree};

mod algorithms;
mod graph;
mod learning;
mod tree;

use algorithms::{ChaodaAlgorithm, ParChaodaAlgorithm};
pub use graph::{Component, Graph, Node};
pub use learning::{Features as AnomalyFeatures, MetaMlModel, metrics, normalize_features};

/// All anomaly detection algorithms provided with CHAODA.
#[derive(Debug, Clone)]
#[must_use]
pub enum Chaoda {
    /// A `Node` is more anomalous if it comes from a cluster whose accumulated cardinality ratio is low.
    AccumulatedCardinalityRatios,
    /// A `Node` is more anomalous if it can reach fewer other nodes in the graph within the same number of steps as compared to other nodes in the graph.
    GraphNeighborhoodSize,
    /// A `Node` is more anomalous if it represents a smaller number of items relative to other `Node`s in the `Graph`.
    RelativeClusterCardinality,
    /// A `Node` is more anomalous if it is in a `Component` whose nodes collectively have fewer items than the other `Components` in the graph.
    RelativeComponentCardinality,
    /// A `Node` is more anomalous if it has fewer neighbors in the graph relative to other nodes in the graph.
    RelativeVertexDegree,
    /// A `Node` is more anomalous if it is visited less frequently during an infinite random walk on the graph.
    StationaryProbabilities,
}

impl Chaoda {
    /// Compute anomaly scores for all items from the tree used to create the graph.
    ///
    /// High scores indicate more anomalous nodes, and low scores indicate less anomalous nodes. The scores are normalized to the range [0, 1] using gaussian
    /// error function normalization.
    ///
    /// # Arguments
    ///
    /// - `graph`: The `Graph` for which to compute anomaly scores.
    /// - `tree`: The `Tree` that was used for creating the `Graph`.
    ///
    /// # Errors
    ///
    /// - If any of the `Cluster`s selected for creating the `Graph` was not found in the `Tree`.
    /// - If the underlying algorithm fails to compute a score for each item in the tree.
    pub fn anomaly_scores<D, M, T, A>(&self, graph: &Graph<T>, tree: &Tree<D, M, T, (A, AnomalyFeatures)>) -> Result<Vec<f64>, String>
    where
        T: DistanceValue,
    {
        match self {
            Self::AccumulatedCardinalityRatios => algorithms::AccumulatedCardinalityRatios.anomaly_scores(graph, tree),
            Self::GraphNeighborhoodSize => algorithms::GraphNeighborhoodSize.anomaly_scores(graph, tree),
            Self::RelativeClusterCardinality => algorithms::RelativeClusterCardinality.anomaly_scores(graph, tree),
            Self::RelativeComponentCardinality => algorithms::RelativeComponentCardinality.anomaly_scores(graph, tree),
            Self::RelativeVertexDegree => algorithms::RelativeVertexDegree.anomaly_scores(graph, tree),
            Self::StationaryProbabilities => algorithms::StationaryProbabilities.anomaly_scores(graph, tree),
        }
    }

    /// Parallel version of [`Self::anomaly_scores`].
    ///
    /// # Errors
    ///
    /// See [`Self::anomaly_scores`] for more details on possible errors.
    pub fn par_anomaly_scores<D, M, T, A>(&self, graph: &Graph<T>, tree: &Tree<D, M, T, (A, AnomalyFeatures)>) -> Result<Vec<f64>, String>
    where
        D: Send + Sync,
        M: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
    {
        match self {
            Self::AccumulatedCardinalityRatios => algorithms::AccumulatedCardinalityRatios.par_anomaly_scores(graph, tree),
            Self::GraphNeighborhoodSize => algorithms::GraphNeighborhoodSize.par_anomaly_scores(graph, tree),
            Self::RelativeClusterCardinality => algorithms::RelativeClusterCardinality.par_anomaly_scores(graph, tree),
            Self::RelativeComponentCardinality => algorithms::RelativeComponentCardinality.par_anomaly_scores(graph, tree),
            Self::RelativeVertexDegree => algorithms::RelativeVertexDegree.par_anomaly_scores(graph, tree),
            Self::StationaryProbabilities => algorithms::StationaryProbabilities.par_anomaly_scores(graph, tree),
        }
    }
}
