//! Utilities for training the Chaoda models.

use distances::Number;

use crate::{adapter::Adapter, Cluster, Dataset, Metric, Partition};

mod algorithms;
mod combination;
mod meta_ml;

pub use algorithms::{GraphAlgorithm, GraphEvaluator};
pub use combination::TrainableCombination;
pub use meta_ml::TrainableMetaMlModel;

use super::{inference::Chaoda, Vertex};

/// A trainer for Chaoda models.
///
/// # Type parameters
///
/// - `I`: The type of the input data.
/// - `U`: The type of the distance values.
/// - `M`: The number of metrics to train with.
pub struct ChaodaTrainer<I, U: Number, const M: usize> {
    /// The distance metrics to train with.
    metrics: [Metric<I, U>; M],
    /// The combinations of `MetaMLModel`s and `GraphAlgorithm`s to train with.
    combinations: [Vec<TrainableCombination>; M],
}

impl<I: Clone, U: Number, const M: usize> ChaodaTrainer<I, U, M> {
    /// Create a new `ChaodaTrainer` with the given metrics and all pairs of
    /// `MetaMLModel`s and `GraphAlgorithm`s.
    ///
    /// # Arguments
    ///
    /// - `metrics`: The distance metrics to train with.
    /// - `meta_ml_models`: The `MetaMLModel`s to train with.
    /// - `graph_algorithms`: The `GraphAlgorithm`s to train with.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_all_pairs(
        metrics: [Metric<I, U>; M],
        meta_ml_models: Vec<TrainableMetaMlModel>,
        graph_algorithms: Vec<GraphAlgorithm>,
    ) -> Self {
        let combinations = meta_ml_models
            .into_iter()
            .flat_map(|meta_ml_model| {
                graph_algorithms.iter().map(move |graph_algorithm| {
                    TrainableCombination::new(meta_ml_model.clone(), graph_algorithm.clone())
                })
            })
            .collect::<Vec<_>>();
        let combinations = metrics
            .iter()
            .map(|_| combinations.clone())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| {
                unreachable!("Could not convert Vec<Vec<TrainableCombination>> to [Vec<TrainableCombination>; {M}]")
            });
        Self { metrics, combinations }
    }

    /// Create a new `ChaodaTrainer` with the given metrics and combinations.
    ///
    /// # Arguments
    ///
    /// - `metrics`: The distance metrics to train with.
    /// - `combinations`: The combinations of `MetaMLModel`s and `GraphAlgorithm`s to train with.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(metrics: [Metric<I, U>; M], combinations: Vec<TrainableCombination>) -> Self {
        let combinations = metrics
            .iter()
            .map(|_| combinations.clone())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| {
                unreachable!("Could not convert Vec<TrainableCombination> to [Vec<TrainableCombination>; {M}]")
            });
        Self { metrics, combinations }
    }

    /// Create trees for use in training.
    ///
    /// # Arguments
    ///
    /// - `datasets`: The datasets to create trees for.
    /// - `criteria`: The criteria to use for creating the trees.
    /// - `seed`: The seed to use for random number generation.
    ///
    /// # Returns
    ///
    /// The trees created from the datasets.
    ///
    /// # Type parameters
    ///
    /// - `N`: The number of datasets to create trees for.
    /// - `D`: The type of the datasets.
    /// - `S`: The type of the `Cluster` to use for creating the `Vertex` trees.
    /// - `C`: The type of the criteria to use for creating the trees.
    pub fn create_trees<const N: usize, D: Dataset<I, U>, S: Partition<I, U, D>, C: Fn(&S) -> bool>(
        &self,
        datasets: &mut [D; N],
        criteria: &[[C; N]; M],
        seed: Option<u64>,
    ) -> [[Vertex<I, U, D, S>; N]; M] {
        let mut trees = Vec::new();
        for (metric, criteria) in self.metrics.iter().zip(criteria) {
            let mut metric_trees = Vec::new();
            for (data, criteria) in datasets.iter_mut().zip(criteria) {
                data.set_metric(metric.clone());
                let root = S::new_tree(data, criteria, seed);
                metric_trees.push(Vertex::adapt_tree(root, None));
            }
            let metric_trees = metric_trees
                .try_into()
                .unwrap_or_else(|_| unreachable!("Could not convert Vec<Vertex> to [Vertex; {M}]"));
            trees.push(metric_trees);
        }
        trees
            .try_into()
            .unwrap_or_else(|_| unreachable!("Could not convert Vec<[Vertex; {M}]> to [[Vertex; {M}]; {N}]"))
    }

    /// Train the model using the given datasets.
    ///
    /// # Arguments
    ///
    /// - `datasets`: The datasets to train with.
    /// - `trees`: The trees built from the datasets.
    /// - `labels`: The labels for the datasets.
    /// - `min_depth`: The minimum depth of `Cluster`s to use for `Graph`s.
    /// - `num_epochs`: The number of epochs to train for.
    ///
    /// # Returns
    ///
    /// The trained model.
    ///
    /// # Errors
    ///
    /// - If the number of labels for a dataset does not match the number of
    ///   data points in that dataset.
    ///
    /// # Type parameters
    ///
    /// - `N`: The number of datasets to train with.
    /// - `D`: The type of the datasets.
    /// - `S`: The type of the `Cluster` that were used to create the `Vertex` trees.
    pub fn train<const N: usize, D: Dataset<I, U>, S: Cluster<I, U, D>>(
        &mut self,
        datasets: &mut [D; N],
        trees: &[[Vertex<I, U, D, S>; N]; M],
        labels: &[Vec<bool>; N],
        min_depth: usize,
        num_epochs: usize,
    ) -> Result<Chaoda<I, U, M>, String> {
        for (data, labels) in datasets.iter().zip(labels.iter()) {
            if data.cardinality() != labels.len() {
                return Err("The number of labels does not match the number of data points.".to_string());
            }
        }
        ftlog::info!("Training with {N} datasets and {M} metrics...");

        let mut trined_combinations = Vec::new();
        for _ in 0..num_epochs {
            // ftlog::info!("Training epoch {}/{num_epochs}", e + 1);

            trined_combinations.clear();
            for ((metric, roots), combinations) in
                self.metrics.iter().zip(trees.iter()).zip(self.combinations.iter_mut())
            {
                trined_combinations.clear();

                for ((data, labels), root) in datasets.iter_mut().zip(labels.iter()).zip(roots.iter()) {
                    data.set_metric(metric.clone());

                    let mut metric_trained_combinations = Vec::new();
                    for combination in combinations.iter_mut() {
                        let mut graph = combination.create_graph(root, data, min_depth);
                        let trained_combination = combination.train_step(&mut graph, labels)?;
                        metric_trained_combinations.push(trained_combination);
                    }
                    trined_combinations.push(metric_trained_combinations);
                }
            }
        }

        let combinations: [_; M] = trined_combinations.try_into().unwrap_or_else(|_| {
            unreachable!("Could not convert Vec<Vec<TrainableCombination>> to [Vec<TrainableCombination>; {M}]")
        });

        Ok(Chaoda::new(self.metrics.clone(), combinations))
    }
}
