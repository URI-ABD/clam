//! Utilities for handling a pair of `MetaMLModel` and `GraphAlgorithm`.

use distances::Number;

use crate::{
    chaoda::{training::GraphEvaluator, Graph, GraphAlgorithm, Vertex},
    cluster::ParCluster,
    dataset::ParDataset,
    metric::ParMetric,
    Cluster, Dataset, Metric,
};

use super::TrainedMetaMlModel;

/// A combination of `TrainedMetaMLModel` and `GraphAlgorithm`.
#[cfg_attr(feature = "disk-io", derive(serde::Serialize, serde::Deserialize))]
#[allow(clippy::module_name_repetitions)]
pub struct TrainedCombination {
    /// The `MetaMLModel` to use.
    meta_ml: TrainedMetaMlModel,
    /// The `GraphAlgorithm` to use.
    graph_algorithm: GraphAlgorithm,
    /// The roc-score achieved during training.
    training_roc_score: f32,
}

impl TrainedCombination {
    /// Get the name of the combination.
    ///
    /// The name is in the format `{meta_ml.short_name()}-{graph_algorithm.name()}`.
    pub fn name(&self) -> String {
        format!("{}-{}", self.meta_ml.short_name(), self.graph_algorithm.name())
    }

    /// Return the roc-score achieved during training.
    #[must_use]
    pub const fn training_roc_score(&self) -> f32 {
        self.training_roc_score
    }

    /// The expected roc-score of the model.
    #[must_use]
    pub fn expected_roc_score(&self) -> f32 {
        if self.invert_scores() {
            1.0 - self.training_roc_score
        } else {
            self.training_roc_score
        }
    }

    /// Whether to invert the prediction scores.
    #[must_use]
    pub fn invert_scores(&self) -> bool {
        self.training_roc_score < 0.5
    }

    /// Whether this combination is not a random coin flip.
    #[must_use]
    pub fn discerns(&self, tol: f32) -> bool {
        let tol = tol.abs().min(0.5);
        (self.training_roc_score - 0.5).abs() > tol
    }

    /// Create a new `TrainedCombination`.
    #[must_use]
    pub const fn new(meta_ml: TrainedMetaMlModel, graph_algorithm: GraphAlgorithm, roc_score: f32) -> Self {
        Self {
            meta_ml,
            graph_algorithm,
            training_roc_score: roc_score,
        }
    }

    /// Get the meta-ML scorer function in a callable for any number of `Vertex`es.
    pub fn meta_ml_scorer<T, S>(&self) -> impl Fn(&[&Vertex<T, S>]) -> Vec<f32> + '_
    where
        T: Number,
        S: Cluster<T>,
    {
        move |clusters| {
            let props = clusters.iter().flat_map(|c| c.ratios()).collect::<Vec<_>>();
            self.meta_ml.predict(&props).unwrap_or_else(|e| unreachable!("{e}"))
        }
    }

    /// Create a `Graph` from the `root` with the given `data` and `min_depth`
    /// using the `TrainedMetaMLModel`.
    pub fn create_graph<'a, I, T, D, M, S>(
        &self,
        root: &'a Vertex<T, S>,
        data: &D,
        metric: &M,
        min_depth: usize,
    ) -> Graph<'a, T, S>
    where
        T: Number,
        D: Dataset<I>,
        M: Metric<I, T>,
        S: Cluster<T>,
    {
        let cluster_scorer = self.meta_ml_scorer();
        Graph::from_root(root, data, metric, cluster_scorer, min_depth)
    }

    /// Predict the anomaly scores of the points in the `data`.
    ///
    /// # Arguments
    ///
    /// * `root`: A root `Vertex` of the tree.
    /// * `data`: The `Dataset` to predict on.
    /// * `min_depth`: The minimum depth at which to consider a `Cluster` for `Graph` construction.
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// * The `Graph` constructed from the `root`.
    /// * The anomaly scores of the points in the `data`.
    pub fn predict<'a, I, T, D, M, S>(
        &self,
        root: &'a Vertex<T, S>,
        data: &D,
        metric: &M,
        min_depth: usize,
    ) -> (Graph<'a, T, S>, Vec<f32>)
    where
        T: Number,
        D: Dataset<I>,
        M: Metric<I, T>,
        S: Cluster<T>,
    {
        ftlog::debug!("Predicting with {}...", self.name());

        let graph = self.create_graph(root, data, metric, min_depth);
        let scores = self.graph_algorithm.evaluate_points(&graph);

        let scores = if self.invert_scores() {
            scores.iter().map(|&s| 1.0 - s).collect()
        } else {
            scores
        };

        (graph, scores)
    }

    /// Parallel version of [`TrainingCombination::create_graph`](crate::chaoda::inference::combination::TrainedCombination::create_graph).
    pub fn par_create_graph<'a, I, T, D, M, S>(
        &self,
        root: &'a Vertex<T, S>,
        data: &D,
        metric: &M,
        min_depth: usize,
    ) -> Graph<'a, T, S>
    where
        I: Send + Sync,
        T: Number,
        D: ParDataset<I>,
        M: ParMetric<I, T>,
        S: ParCluster<T>,
    {
        let cluster_scorer = self.meta_ml_scorer();
        Graph::par_from_root(root, data, metric, cluster_scorer, min_depth)
    }

    /// Parallel version of [`TrainingCombination::predict`](crate::chaoda::inference::combination::TrainedCombination::predict).
    pub fn par_predict<'a, I, T, D, M, S>(
        &self,
        root: &'a Vertex<T, S>,
        data: &D,
        metric: &M,
        min_depth: usize,
    ) -> (Graph<'a, T, S>, Vec<f32>)
    where
        I: Send + Sync,
        T: Number,
        D: ParDataset<I>,
        M: ParMetric<I, T>,
        S: ParCluster<T>,
    {
        ftlog::debug!("Predicting with {}...", self.name());

        let graph = self.par_create_graph(root, data, metric, min_depth);
        let scores = self.graph_algorithm.evaluate_points(&graph);

        let scores = if self.invert_scores() {
            scores.iter().map(|&s| 1.0 - s).collect()
        } else {
            scores
        };

        (graph, scores)
    }
}
