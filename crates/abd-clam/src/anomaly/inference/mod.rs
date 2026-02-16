//! Utilities for running inference with pre-trained Chaoda models.
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::{Dataset, DistanceValue, ParCluster, ParDataset, ParPartition, Partition};

use super::{roc_auc_score, OddBall};

mod combination;
mod meta_ml;
mod trained_smc;

pub use combination::TrainedCombination;
pub use meta_ml::TrainedMetaMlModel;
pub use trained_smc::TrainedSmc;

use super::Metrics;

/// A pre-trained Chaoda model.
#[must_use]
pub struct Chaoda<'m, I, T: DistanceValue, const M: usize> {
    /// The distance metrics to train with.
    metrics: Metrics<'m, I, T, M>,
    /// The trained models.
    combinations: [Vec<TrainedCombination>; M],
}

impl<'m, I: Clone + Send + Sync, T: DistanceValue + Send + Sync, const M: usize> Chaoda<'m, I, T, M> {
    /// Create a new Chaoda model with the given metrics and trained combinations.
    pub const fn new(metrics: Metrics<'m, I, T, M>, combinations: [Vec<TrainedCombination>; M]) -> Self {
        Self { metrics, combinations }
    }

    /// Get the distance metrics used by the model.
    #[must_use]
    pub const fn metrics(&self) -> &Metrics<'m, I, T, M> {
        &self.metrics
    }

    /// Set the distance metrics to be used by the model.
    pub fn set_metrics(&mut self, metrics: Metrics<'m, I, T, M>) {
        self.metrics = metrics;
    }

    /// Create trees to use for inference, one for each metric.
    pub fn create_trees<D: Dataset<I>, S: Partition<T>, C: Fn(&S) -> bool>(
        &self,
        data: &D,
        criteria: &[C; M],
    ) -> [OddBall<T, S>; M] {
        let mut trees = Vec::new();
        for (metric, criteria) in self.metrics.iter().zip(criteria.iter()) {
            let source = S::new_tree(data, metric, criteria);
            let tree = OddBall::from_cluster_tree(source);
            trees.push(tree);
        }
        trees
            .try_into()
            .unwrap_or_else(|_| unreachable!("Could not convert Vec<OddBall<I, U, D, S>> to [OddBall<I, U, D, S>; {M}]"))
    }

    /// Run inference on the given data.
    pub fn predict_from_trees<D: Dataset<I>, S: Partition<T>>(
        &self,
        data: &D,
        trees: &[OddBall<T, S>; M],
        min_depth: usize,
    ) -> Vec<f32> {
        // TODO: Make this a parameter.
        let tol = 0.01;

        let mut num_discerning = 0;
        let mut scores = Vec::new();
        for ((metric, root), combinations) in self.metrics.iter().zip(trees.iter()).zip(self.combinations.iter()) {
            for c in combinations {
                if c.discerns(tol) {
                    num_discerning += 1;
                    let (_, row) = c.predict(root, data, metric, min_depth);
                    scores.extend_from_slice(&row);
                }
            }
        }

        let shape = (data.cardinality(), num_discerning);
        let scores_len = scores.len();
        let scores = Array2::from_shape_vec(shape, scores).unwrap_or_else(|e| {
            unreachable!(
                "Could not create Array2 of shape {shape:?} from Vec<f32> of len {}: {e}",
                scores_len
            )
        });
        scores
            .mean_axis(Axis(1))
            .unwrap_or_else(|| unreachable!("Could not compute mean of Array2<f32> along axis 1"))
            .to_vec()
    }

    /// Run inference on the given data.
    pub fn predict<D: Dataset<I>, S: Partition<T>, C: Fn(&S) -> bool>(
        &self,
        data: &D,
        criteria: &[C; M],
        min_depth: usize,
    ) -> Vec<f32> {
        let trees = self.create_trees(data, criteria);
        self.predict_from_trees(data, &trees, min_depth)
    }

    /// Evaluate the model on the given data.
    pub fn evaluate<D: Dataset<I>, S: Partition<T>, C: Fn(&S) -> bool>(
        &self,
        data: &D,
        criteria: &[C; M],
        labels: &[bool],
        min_depth: usize,
    ) -> f32 {
        let scores = self.predict(data, criteria, min_depth);
        roc_auc_score(labels, &scores)
            .unwrap_or_else(|e| unreachable!("Could not compute ROC-AUC score for evaluation: {e}"))
    }
}

impl<I: Clone + Send + Sync, T: DistanceValue + Send + Sync, const M: usize> Chaoda<'_, I, T, M> {
    /// Parallel version of [`Chaoda::create_trees`](crate::chaoda::Chaoda::create_trees).
    pub fn par_create_trees<D: ParDataset<I>, S: ParPartition<T>, C: (Fn(&S) -> bool) + Send + Sync>(
        &self,
        data: &D,
        criteria: &[C; M],
    ) -> [OddBall<T, S>; M] {
        let mut trees = Vec::new();
        for (metric, criteria) in self.metrics.iter().zip(criteria.iter()) {
            let source = S::par_new_tree(data, metric, criteria);
            let tree = OddBall::from_cluster_tree(source);
            trees.push(tree);
        }
        trees
            .try_into()
            .unwrap_or_else(|_| unreachable!("Could not convert Vec<OddBall<I, U, D, S>> to [OddBall<I, U, D, S>; {M}]"))
    }

    /// Parallel version of [`Chaoda::predict_from_trees`](crate::chaoda::Chaoda::predict_from_trees).
    pub fn par_predict_from_trees<D: ParDataset<I>, S: ParCluster<T>>(
        &self,
        data: &D,
        trees: &[OddBall<T, S>; M],
        min_depth: usize,
    ) -> Vec<f32> {
        // TODO: Make this a parameter.
        let tol = 0.05;

        let mut num_discerning = 0;
        let mut scores = Vec::new();

        for ((metric, root), combinations) in self.metrics.iter().zip(trees.iter()).zip(self.combinations.iter()) {
            let new_scores = combinations
                .par_iter()
                .filter_map(|c| {
                    if c.discerns(tol) {
                        let (_, row) = c.predict(root, data, metric, min_depth);
                        Some(row)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            num_discerning += new_scores.len();
            scores.extend(new_scores.into_iter().flatten());
        }

        let shape = (data.cardinality(), num_discerning);
        let scores_len = scores.len();
        let scores = Array2::from_shape_vec(shape, scores).unwrap_or_else(|e| {
            unreachable!(
                "Could not create Array2 of shape {shape:?} from Vec<f32> of len {}: {e}",
                scores_len
            )
        });
        scores
            .mean_axis(Axis(1))
            .unwrap_or_else(|| unreachable!("Could not compute mean of Array2<f32> along axis 1"))
            .to_vec()
    }

    /// Parallel version of [`Chaoda::predict`](crate::chaoda::Chaoda::predict).
    pub fn par_predict<D: ParDataset<I>, S: ParPartition<T>, C: (Fn(&S) -> bool) + Send + Sync>(
        &self,
        data: &D,
        criteria: &[C; M],
        min_depth: usize,
    ) -> Vec<f32> {
        let trees = self.par_create_trees(data, criteria);
        self.par_predict_from_trees(data, &trees, min_depth)
    }

    /// Parallel version of [`Chaoda::evaluate`](crate::chaoda::Chaoda::evaluate).
    pub fn par_evaluate<D: ParDataset<I>, S: ParPartition<T>, C: (Fn(&S) -> bool) + Send + Sync>(
        &self,
        data: &mut D,
        criteria: &[C; M],
        labels: &[bool],
        min_depth: usize,
    ) -> f32 {
        let scores = self.par_predict(data, criteria, min_depth);
        roc_auc_score(labels, &scores)
            .unwrap_or_else(|e| unreachable!("Could not compute ROC-AUC score for evaluation: {e}"))
    }
}
