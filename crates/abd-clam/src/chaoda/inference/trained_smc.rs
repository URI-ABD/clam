//! A trained Single-Metric-CHAODA ensemble.
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::{
    chaoda::{roc_auc_score, OddBall, ParVertex, Vertex},
    Dataset, DistanceValue, ParDataset, ParPartition, Partition,
};

use super::TrainedCombination;

/// A trained Single-Metric-CHAODA ensemble.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TrainedSmc(Vec<TrainedCombination>);

impl TrainedSmc {
    /// Create a new trained Single-Metric-CHAODA ensemble.
    ///
    /// # Arguments
    ///
    /// - `combinations`: The trained combinations to use.
    ///
    /// # Returns
    ///
    /// The trained Single-Metric-CHAODA ensemble.
    #[must_use]
    pub const fn new(combinations: Vec<TrainedCombination>) -> Self {
        Self(combinations)
    }

    /// Get the trained combinations.
    #[must_use]
    pub const fn combinations(&self) -> &Vec<TrainedCombination> {
        &self.0
    }

    /// Create a tree for inference.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset to use.
    /// - `metric`: The metric to use.
    /// - `criteria`: The criteria to use for partitioning.
    /// - `seed`: The seed to use for random number generation.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The type of the dataset items.
    /// - `T`: The type of the metric values.
    /// - `D`: The type of the dataset.
    /// - `M`: The type of the metric.
    /// - `S`: The type of the cluster that will be adapted to `OddBall`.
    /// - `C`: The type of the criteria function.
    fn create_tree<I, T, D, M, S, C>(data: &D, metric: &M, criteria: &C) -> OddBall<T, S>
    where
        T: DistanceValue,
        D: Dataset<I>,
        M: Fn(&I, &I) -> T,
        S: Partition<T>,
        C: Fn(&S) -> bool,
    {
        let source = S::new_tree(data, metric, criteria);
        OddBall::from_cluster_tree(source)
    }

    /// Run inference on the given data using the pre-built tree.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset to use.
    /// - `metric`: The metric to use.
    /// - `root`: The root of the tree to use.
    /// - `min_depth`: The minimum depth to consider for selecting clusters to
    ///   create graphs.
    /// - `tol`: The tolerance to use for discerning meta-ml models. This should
    ///   be a small positive number, ideally less than `0.1`.
    ///
    /// # Returns
    ///
    /// The predicted anomaly scores for each item in the dataset.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The type of the dataset items.
    /// - `T`: The type of the metric values.
    /// - `D`: The type of the dataset.
    /// - `M`: The type of the metric.
    /// - `V`: The type of the vertex in the tree.
    pub fn predict_from_tree<I, T, D, M, V>(
        &self,
        data: &D,
        metric: &M,
        root: &V,
        min_depth: usize,
        tol: f32,
    ) -> Vec<f32>
    where
        T: DistanceValue,
        D: Dataset<I>,
        M: Fn(&I, &I) -> T,
        V: Vertex<T>,
    {
        let (num_discerning, scores) = self
            .0
            .iter()
            .enumerate()
            .filter(|(_, combination)| combination.discerns(tol))
            .fold((0, Vec::new()), |(num_discerning, mut scores), (i, combination)| {
                ftlog::info!(
                    "Predicting with combination {}/{} {}",
                    i + 1,
                    self.0.len(),
                    combination.name()
                );
                let (_, mut row) = combination.predict(root, data, metric, min_depth);
                scores.append(&mut row);
                (num_discerning + 1, scores)
            });

        if num_discerning == 0 {
            ftlog::warn!("No discerning combinations found. Returning all scores as `0.5`.");
            return vec![0.5; data.cardinality()];
        }

        ftlog::info!("Averaging scores from {num_discerning} discerning combinations.");
        let shape = (data.cardinality(), num_discerning);
        let scores_len = scores.len();
        let scores = Array2::from_shape_vec(shape, scores).unwrap_or_else(|e| {
            unreachable!(
                "Could not convert Vec<T> of len {scores_len} to Array2<T> of shape {:?}: {e}",
                shape
            )
        });

        scores
            .mean_axis(Axis(1))
            .unwrap_or_else(|| unreachable!("Could not compute mean of Array2<T> along axis 1"))
            .to_vec()
    }

    /// Run inference on the given data.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset to use.
    /// - `metric`: The metric to use.
    /// - `criteria`: The criteria to use for partitioning.
    /// - `seed`: The seed to use for random number generation.
    /// - `min_depth`: The minimum depth to consider for selecting clusters to
    ///   create graphs.
    /// - `tol`: The tolerance to use for discerning meta-ml models. This should
    ///   be a small positive number, ideally less than `0.1`.
    ///
    /// # Returns
    ///
    /// The predicted anomaly scores for each item in the dataset.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The type of the dataset items.
    /// - `T`: The type of the metric values.
    /// - `D`: The type of the dataset.
    /// - `M`: The type of the metric.
    /// - `S`: The type of the cluster that will be adapted to `OddBall`.
    /// - `C`: The type of the criteria function.
    pub fn predict<I, T, D, M, S, C>(&self, data: &D, metric: &M, criteria: &C, min_depth: usize, tol: f32) -> Vec<f32>
    where
        T: DistanceValue,
        D: Dataset<I>,
        M: Fn(&I, &I) -> T,
        S: Partition<T>,
        C: Fn(&S) -> bool,
    {
        let root = Self::create_tree(data, metric, criteria);
        self.predict_from_tree(data, metric, &root, min_depth, tol)
    }

    /// Evaluate the model on the given data.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset to use.
    /// - `labels`: The labels to use for evaluation.
    /// - `metric`: The metric to use.
    /// - `criteria`: The criteria to use for partitioning.
    /// - `seed`: The seed to use for random number generation.
    /// - `min_depth`: The minimum depth to consider for selecting clusters to
    ///   create graphs.
    /// - `tol`: The tolerance to use for discerning meta-ml models. This should
    ///   be a small positive number, ideally less than `0.1`.
    ///
    /// # Returns
    ///
    /// The predicted anomaly scores for each item in the dataset.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The type of the dataset items.
    /// - `T`: The type of the metric values.
    /// - `D`: The type of the dataset.
    /// - `M`: The type of the metric.
    /// - `S`: The type of the cluster that will be adapted to `OddBall`.
    /// - `C`: The type of the criteria function.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<I, T, D, M, S, C>(
        &self,
        data: &D,
        labels: &[bool],
        metric: &M,
        criteria: &C,
        min_depth: usize,
        tol: f32,
    ) -> f32
    where
        T: DistanceValue,
        D: Dataset<I>,
        M: Fn(&I, &I) -> T,
        S: Partition<T>,
        C: Fn(&S) -> bool,
    {
        let root = Self::create_tree(data, metric, criteria);
        let scores = self.predict_from_tree(data, metric, &root, min_depth, tol);
        roc_auc_score(labels, &scores)
            .unwrap_or_else(|e| unreachable!("Could not compute ROC-AUC score for dataset: {e}"))
    }

    /// Parallel version of [`TrainedSmc::create_tree`](crate::chaoda::inference::TrainedSmc::create_tree).
    fn par_create_tree<I, T, D, M, S, C>(data: &D, metric: &M, criteria: &C) -> OddBall<T, S>
    where
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        S: ParPartition<T>,
        C: (Fn(&S) -> bool) + Send + Sync,
    {
        let source = S::par_new_tree(data, metric, criteria);
        OddBall::from_cluster_tree(source)
    }

    /// Parallel version of [`TrainedSmc::predict_from_tree`](crate::chaoda::inference::TrainedSmc::predict_from_tree).
    pub fn par_predict_from_tree<I, T, D, M, V>(
        &self,
        data: &D,
        metric: &M,
        root: &V,
        min_depth: usize,
        tol: f32,
    ) -> Vec<f32>
    where
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        V: ParVertex<T>,
    {
        let (num_discerning, scores) = self
            .0
            .par_iter()
            .enumerate()
            .filter(|(_, combination)| combination.discerns(tol))
            .fold(
                || (0, Vec::new()),
                |(num_discerning, mut scores), (i, combination)| {
                    ftlog::info!(
                        "Predicting with combination {}/{} {}",
                        i + 1,
                        self.0.len(),
                        combination.name()
                    );
                    let (_, mut row) = combination.par_predict(root, data, metric, min_depth);
                    scores.append(&mut row);
                    (num_discerning + 1, scores)
                },
            )
            .reduce(
                || (0, Vec::new()),
                |(num_discerning, mut scores), (n, s)| {
                    scores.extend(s);
                    (num_discerning + n, scores)
                },
            );

        if num_discerning == 0 {
            ftlog::warn!("No discerning combinations found. Returning all scores as `0.5`.");
            return vec![0.5; data.cardinality()];
        }

        ftlog::info!("Averaging scores from {num_discerning} discerning combinations.");
        let shape = (data.cardinality(), num_discerning);
        let scores_len = scores.len();
        let scores = Array2::from_shape_vec(shape, scores).unwrap_or_else(|e| {
            unreachable!(
                "Could not convert Vec<T> of len {scores_len} to Array2<T> of shape {:?}: {e}",
                shape
            )
        });

        scores
            .mean_axis(Axis(1))
            .unwrap_or_else(|| unreachable!("Could not compute mean of Array2<T> along axis 1"))
            .to_vec()
    }

    /// Parallel version of [`TrainedSmc::predict`](crate::chaoda::inference::TrainedSmc::predict).
    pub fn par_predict<I, T, D, M, S, C>(
        &self,
        data: &D,
        metric: &M,
        criteria: &C,
        min_depth: usize,
        tol: f32,
    ) -> Vec<f32>
    where
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        S: ParPartition<T>,
        C: (Fn(&S) -> bool) + Send + Sync,
    {
        let root = Self::par_create_tree(data, metric, criteria);
        self.par_predict_from_tree(data, metric, &root, min_depth, tol)
    }

    /// Parallel version of [`TrainedSmc::evaluate`](crate::chaoda::inference::TrainedSmc::evaluate).
    #[allow(clippy::too_many_arguments)]
    pub fn par_evaluate<I, T, D, M, S, C>(
        &self,
        data: &D,
        labels: &[bool],
        metric: &M,
        criteria: &C,
        min_depth: usize,
        tol: f32,
    ) -> f32
    where
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        S: ParPartition<T>,
        C: (Fn(&S) -> bool) + Send + Sync,
    {
        let root = Self::par_create_tree(data, metric, criteria);
        let scores = self.par_predict_from_tree(data, metric, &root, min_depth, tol);
        roc_auc_score(labels, &scores)
            .unwrap_or_else(|e| unreachable!("Could not compute ROC-AUC score for dataset: {e}"))
    }
}
