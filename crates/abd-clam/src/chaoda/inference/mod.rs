//! Utilities for running inference with pre-trained Chaoda models.

use distances::Number;
use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    adapter::{Adapter, ParAdapter},
    cluster::ParCluster,
    dataset::ParDataset,
    partition::ParPartition,
    Dataset, Metric, Partition,
};

use super::{roc_auc_score, Vertex};

mod combination;
mod meta_ml;

pub use combination::TrainedCombination;
pub use meta_ml::TrainedMetaMlModel;

/// A pre-trained Chaoda model.
#[derive(Serialize, Deserialize)]
pub struct Chaoda<I, U: Number, const M: usize> {
    /// The distance metrics to train with.
    #[serde(with = "serde_arrays")]
    metrics: [Metric<I, U>; M],
    /// The trained models.
    #[serde(with = "serde_arrays")]
    combinations: [Vec<TrainedCombination>; M],
}

impl<I: Clone, U: Number, const M: usize> Chaoda<I, U, M> {
    /// Create a new Chaoda model with the given metrics and trained combinations.
    #[must_use]
    pub const fn new(metrics: [Metric<I, U>; M], combinations: [Vec<TrainedCombination>; M]) -> Self {
        Self { metrics, combinations }
    }

    /// Get the distance metrics used by the model.
    #[must_use]
    pub const fn metrics(&self) -> &[Metric<I, U>; M] {
        &self.metrics
    }

    /// Set the distance metrics to be used by the model.
    pub fn set_metrics(&mut self, metrics: [Metric<I, U>; M]) {
        self.metrics = metrics;
    }

    /// Create trees to use for inference, one for each metric.
    pub fn create_trees<D: Dataset<I, U>, S: Partition<I, U, D>, C: Fn(&S) -> bool>(
        &self,
        data: &mut D,
        criteria: &[C; M],
        seed: Option<u64>,
    ) -> [Vertex<I, U, D, S>; M] {
        let mut trees = Vec::new();
        for (metric, criteria) in self.metrics.iter().zip(criteria.iter()) {
            data.set_metric(metric.clone());
            let source = S::new_tree(data, criteria, seed);
            let tree = Vertex::adapt_tree(source, None);
            trees.push(tree);
        }
        trees
            .try_into()
            .unwrap_or_else(|_| unreachable!("Could not convert Vec<Vertex<I, U, D, S>> to [Vertex<I, U, D, S>; {M}]"))
    }

    /// Run inference on the given data.
    pub fn predict_from_trees<D: Dataset<I, U>, S: Partition<I, U, D>>(
        &self,
        data: &mut D,
        trees: &[Vertex<I, U, D, S>; M],
        min_depth: usize,
    ) -> Vec<f32> {
        // TODO: Make this a parameter.
        let tol = 0.01;

        let mut num_discerning = 0;
        let mut scores = Vec::new();
        for ((metric, root), combinations) in self.metrics.iter().zip(trees.iter()).zip(self.combinations.iter()) {
            data.set_metric(metric.clone());
            for c in combinations {
                if c.discerns(tol) {
                    num_discerning += 1;
                    let (_, row) = c.predict(root, data, min_depth);
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
    pub fn predict<D: Dataset<I, U>, S: Partition<I, U, D>, C: Fn(&S) -> bool>(
        &self,
        data: &mut D,
        criteria: &[C; M],
        seed: Option<u64>,
        min_depth: usize,
    ) -> Vec<f32> {
        let trees = self.create_trees(data, criteria, seed);
        self.predict_from_trees(data, &trees, min_depth)
    }

    /// Evaluate the model on the given data.
    pub fn evaluate<D: Dataset<I, U>, S: Partition<I, U, D>, C: Fn(&S) -> bool>(
        &self,
        data: &mut D,
        criteria: &[C; M],
        labels: &[bool],
        seed: Option<u64>,
        min_depth: usize,
    ) -> f32 {
        let scores = self.predict(data, criteria, seed, min_depth);
        roc_auc_score(labels, &scores)
            .unwrap_or_else(|e| unreachable!("Could not compute ROC-AUC score for evaluation: {e}"))
    }
}

impl<I: Clone + Send + Sync, U: Number, const M: usize> Chaoda<I, U, M> {
    /// Parallel version of `create_trees`.
    pub fn par_create_trees<D: ParDataset<I, U>, S: ParPartition<I, U, D>, C: (Fn(&S) -> bool) + Send + Sync>(
        &self,
        data: &mut D,
        criteria: &[C; M],
        seed: Option<u64>,
    ) -> [Vertex<I, U, D, S>; M] {
        let mut trees = Vec::new();
        for (metric, criteria) in self.metrics.iter().zip(criteria.iter()) {
            data.set_metric(metric.clone());
            let source = S::par_new_tree(data, criteria, seed);
            let tree = Vertex::par_adapt_tree(source, None);
            trees.push(tree);
        }
        trees
            .try_into()
            .unwrap_or_else(|_| unreachable!("Could not convert Vec<Vertex<I, U, D, S>> to [Vertex<I, U, D, S>; {M}]"))
    }

    /// Run inference on the given data.
    pub fn par_predict_from_trees<D: ParDataset<I, U>, S: ParCluster<I, U, D>>(
        &self,
        data: &mut D,
        trees: &[Vertex<I, U, D, S>; M],
        min_depth: usize,
    ) -> Vec<f32> {
        // TODO: Make this a parameter.
        let tol = 0.05;

        let mut num_discerning = 0;
        let mut scores = Vec::new();

        for ((metric, root), combinations) in self.metrics.iter().zip(trees.iter()).zip(self.combinations.iter()) {
            data.set_metric(metric.clone());
            let new_scores = combinations
                .par_iter()
                .filter_map(|c| {
                    if c.discerns(tol) {
                        let (_, row) = c.predict(root, data, min_depth);
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

    /// Parallel version of `predict`.
    pub fn par_predict<D: ParDataset<I, U>, S: ParPartition<I, U, D>, C: (Fn(&S) -> bool) + Send + Sync>(
        &self,
        data: &mut D,
        criteria: &[C; M],
        seed: Option<u64>,
        min_depth: usize,
    ) -> Vec<f32> {
        let trees = self.par_create_trees(data, criteria, seed);
        self.par_predict_from_trees(data, &trees, min_depth)
    }

    /// Parallel version of `evaluate`.
    pub fn par_evaluate<D: ParDataset<I, U>, S: ParPartition<I, U, D>, C: (Fn(&S) -> bool) + Send + Sync>(
        &self,
        data: &mut D,
        criteria: &[C; M],
        labels: &[bool],
        seed: Option<u64>,
        min_depth: usize,
    ) -> f32 {
        let scores = self.par_predict(data, criteria, seed, min_depth);
        roc_auc_score(labels, &scores)
            .unwrap_or_else(|e| unreachable!("Could not compute ROC-AUC score for evaluation: {e}"))
    }
}
