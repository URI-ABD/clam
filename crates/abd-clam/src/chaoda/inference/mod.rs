//! Utilities for running inference with pre-trained Chaoda models.

mod combination;
mod meta_ml;

use distances::Number;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{adapter::Adapter, Dataset, Metric, Partition};

pub use combination::TrainedCombination;
pub use meta_ml::TrainedMetaMlModel;

use super::Vertex;

/// A pre-trained Chaoda model.
#[derive(Clone)]
pub struct Chaoda<I, U: Number, const M: usize> {
    /// The distance metrics to train with.
    metrics: [Metric<I, U>; M],
    /// The trained models.
    combinations: [Vec<TrainedCombination>; M],
}

impl<I: Clone, U: Number, const M: usize> Chaoda<I, U, M> {
    /// Create a new Chaoda model with the given metrics and trained combinations.
    pub fn new(metrics: [Metric<I, U>; M], combinations: [Vec<TrainedCombination>; M]) -> Self {
        Self { metrics, combinations }
    }

    /// Get the distance metrics used by the model.
    pub fn metrics(&self) -> &[Metric<I, U>; M] {
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
        let mut scores = Vec::new();
        for ((metric, root), combinations) in self.metrics.iter().zip(trees.iter()).zip(self.combinations.iter()) {
            data.set_metric(metric.clone());
            for c in combinations {
                let (_, row) = c.predict(root, data, min_depth);
                scores.extend_from_slice(&row);
            }
        }
        let shape = (data.cardinality(), M);
        let scores = Array2::from_shape_vec(shape, scores)
            .unwrap_or_else(|e| unreachable!("Could not create Array2 of shape {shape:?} from Vec<f32>: {e}"));
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
}

#[derive(Serialize, Deserialize)]
struct ChaodaSerde<I, U: Number, const M: usize> {
    combinations: Vec<Vec<TrainedCombination>>,
    _p: std::marker::PhantomData<(I, U)>,
}

impl<I: Clone, U: Number, const M: usize> ChaodaSerde<I, U, M> {
    /// Create a new `ChaodaSerde` from a `Chaoda`.
    fn from_chaoda(chaoda: &Chaoda<I, U, M>) -> Self {
        Self {
            combinations: chaoda.combinations.clone().to_vec(),
            _p: std::marker::PhantomData,
        }
    }

    /// Create a `Chaoda` from a `ChaodaSerde`.
    fn into_chaoda(self) -> Chaoda<I, U, M> {
        let metrics = vec![Metric::default(); M]
            .try_into()
            .unwrap_or_else(|_| unreachable!("Could not convert Vec of default metrics into array."));
        let combinations = self
            .combinations
            .try_into()
            .unwrap_or_else(|_| unreachable!("Could not convert Vec of TrainedCombinations into array."));
        Chaoda::new(metrics, combinations)
    }
}

impl<I: Clone, U: Number, const M: usize> Serialize for Chaoda<I, U, M> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ChaodaSerde::from_chaoda(self).serialize(serializer)
    }
}

impl<'de, I: Clone, U: Number, const M: usize> Deserialize<'de> for Chaoda<I, U, M> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ChaodaSerde::deserialize(deserializer).map(ChaodaSerde::into_chaoda)
    }
}
