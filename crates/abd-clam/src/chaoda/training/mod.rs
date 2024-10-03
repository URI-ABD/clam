//! Utilities for training the Chaoda models.

use distances::Number;
use rayon::prelude::*;

use crate::{
    adapter::{Adapter, ParAdapter},
    chaoda::inference::TrainedCombination,
    cluster::ParCluster,
    dataset::ParDataset,
    partition::ParPartition,
    Cluster, Dataset, Metric, Partition,
};

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
        criteria: &[[C; M]; N],
        seed: Option<u64>,
    ) -> [[Vertex<I, U, D, S>; M]; N] {
        let mut trees = Vec::new();
        for (data, criteria) in datasets.iter_mut().zip(criteria) {
            let mut metric_trees = Vec::new();
            for (metric, criteria) in self.metrics.iter().zip(criteria) {
                data.set_metric(metric.clone());
                let root = S::new_tree(data, criteria, seed);
                metric_trees.push(Vertex::adapt_tree(root, None));
            }
            let metric_trees = metric_trees.try_into().unwrap_or_else(|v: Vec<_>| {
                unreachable!("Could not convert Vec<Vertex> to [Vertex; {M}]. Len was {}", v.len())
            });
            trees.push(metric_trees);
        }
        trees.try_into().unwrap_or_else(|v: Vec<_>| {
            unreachable!(
                "Could not convert Vec<[Vertex; {M}]> to [[Vertex; {M}]; {N}]. Len was {}",
                v.len()
            )
        })
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
        trees: &[[Vertex<I, U, D, S>; M]; N],
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

        let mut trained_combinations = Vec::new();

        for e in 0..num_epochs {
            ftlog::info!("Starting Training epoch {}/{num_epochs}", e + 1);
            trained_combinations.clear();

            for (i, ((data, labels), roots)) in datasets.iter_mut().zip(labels).zip(trees.iter()).enumerate() {
                let mut data_combinations = Vec::new();

                for (j, ((metric, root), combinations)) in self
                    .metrics
                    .iter()
                    .zip(roots.iter())
                    .zip(self.combinations.iter_mut())
                    .enumerate()
                {
                    data.set_metric(metric.clone());

                    let mut inner_combinations = Vec::new();
                    let n_combos = combinations.len();
                    for (k, combination) in combinations.iter_mut().enumerate() {
                        let graph = combination.create_graph(root, data, min_depth);
                        let ([x, y], roc_score) = combination.data_from_graph(&graph, labels)?;
                        combination.append_data(&x, &y, Some(roc_score))?;
                        let trained_combination = combination.train_step()?;
                        ftlog::info!(
                            "Epoch {}/{num_epochs}: Data {}/{N}, metric {}/{M}, combination {}: {}/{}, roc-auc: {:.6}",
                            e + 1,
                            i + 1,
                            j + 1,
                            trained_combination.name(),
                            k + 1,
                            n_combos,
                            trained_combination.training_roc_score(),
                        );
                        inner_combinations.push(trained_combination);
                    }
                    data_combinations.push(inner_combinations);
                }

                trained_combinations = data_combinations;
            }

            let roc_scores = trained_combinations
                .iter()
                .flat_map(|combos| combos.iter().map(TrainedCombination::training_roc_score))
                .collect::<Vec<_>>();
            let mean_roc_score: f32 = crate::utils::mean(&roc_scores);
            ftlog::info!(
                "Finished Training epoch {}/{num_epochs} with mean roc-auc: {mean_roc_score}",
                e + 1
            );
        }

        let combinations: [_; M] = trained_combinations.try_into().unwrap_or_else(|v: Vec<_>| {
            unreachable!(
                "Could not convert Vec<Vec<TrainableCombination>> to [Vec<TrainableCombination>; {M}], len: {}",
                v.len()
            )
        });

        for (i, (a, b)) in combinations.iter().zip(self.combinations.iter()).enumerate() {
            if a.len() != b.len() {
                return Err(format!("Mismatch in number of trained combinations for metric {i}"));
            }
        }

        Ok(Chaoda::new(self.metrics.clone(), combinations))
    }
}

impl<I: Clone + Send + Sync, U: Number, const M: usize> ChaodaTrainer<I, U, M> {
    /// Parallel version of `create_trees`.
    pub fn par_create_trees<
        const N: usize,
        D: ParDataset<I, U>,
        S: ParPartition<I, U, D>,
        C: (Fn(&S) -> bool) + Send + Sync,
    >(
        &self,
        datasets: &mut [D; N],
        criteria: &[[C; M]; N],
        seed: Option<u64>,
    ) -> [[Vertex<I, U, D, S>; M]; N] {
        let mut trees = Vec::new();
        for (data, criteria) in datasets.iter_mut().zip(criteria) {
            let mut metric_trees = Vec::new();
            for (metric, criteria) in self.metrics.iter().zip(criteria) {
                data.set_metric(metric.clone());
                let root = S::par_new_tree(data, criteria, seed);
                metric_trees.push(Vertex::par_adapt_tree(root, None));
            }
            let metric_trees = metric_trees.try_into().unwrap_or_else(|v: Vec<_>| {
                unreachable!("Could not convert Vec<Vertex> to [Vertex; {M}]. Len was {}", v.len())
            });
            trees.push(metric_trees);
        }
        trees.try_into().unwrap_or_else(|v: Vec<_>| {
            unreachable!(
                "Could not convert Vec<[Vertex; {M}]> to [[Vertex; {M}]; {N}]. Len was {}",
                v.len()
            )
        })
    }

    /// Parallel version of `train`.
    ///
    /// # Errors
    ///
    /// See `ChaodaTrainer::train`.
    pub fn par_train<const N: usize, D: ParDataset<I, U>, S: ParCluster<I, U, D>>(
        &mut self,
        datasets: &mut [D; N],
        trees: &[[Vertex<I, U, D, S>; M]; N],
        labels: &[Vec<bool>; N],
        min_depth: usize,
        num_epochs: usize,
    ) -> Result<Chaoda<I, U, M>, String> {
        // The only parallelism here is across combinations and in graph
        // creation. This is because the dataset needs to be mutable to change
        // the metric for each pair of dataset and metric.
        for (data, labels) in datasets.iter().zip(labels.iter()) {
            if data.cardinality() != labels.len() {
                return Err("The number of labels does not match the number of data points.".to_string());
            }
        }
        ftlog::info!("Training with {N} datasets and {M} metrics...");

        let mut trained_combinations = Vec::new();

        for e in 0..num_epochs {
            ftlog::info!("Starting Training epoch {}/{num_epochs}", e + 1);

            trained_combinations = datasets
                .iter_mut()
                .zip(labels)
                .zip(trees.iter())
                .enumerate()
                .map(|(i, ((data, labels), roots))| {
                    self.metrics
                        .iter_mut()
                        .zip(roots.iter())
                        .zip(self.combinations.iter_mut())
                        .enumerate()
                        .inspect(|(j, (_, combinations))| {
                            let num_samples = combinations.iter().map(|c| c.train_y.len()).sum::<usize>();
                            ftlog::info!(
                                "Training Epoch {}/{num_epochs}: Data {}/{N}, metric {}/{M}, num_combinations: {}, num_samples: {num_samples}",
                                e + 1,
                                i + 1,
                                j + 1,
                                combinations.len(),
                            );
                        })
                        .map(|(j, ((metric, root), combinations))| {
                            data.set_metric(metric.clone());
                            // Parallelism across combinations and in graph creation
                            let dm_combinations = combinations
                                .par_iter_mut()
                                .map(|combination| {
                                    let graph = combination.par_create_graph(root, data, min_depth);
                                    let ([x, y], roc_score) = combination.data_from_graph(&graph, labels)?;
                                    combination.append_data(&x, &y, Some(roc_score))?;
                                    combination.train_step()
                                })
                                .collect::<Result<Vec<_>, _>>();
                            (j, dm_combinations)
                        })
                        .inspect(|(j, v)| {
                            if let Ok(dm_combinations) = v {
                                let roc_scores = dm_combinations
                                    .iter()
                                    .map(TrainedCombination::expected_roc_score)
                                    .collect::<Vec<_>>();
                                let mean_roc_score: f32 = crate::utils::mean(&roc_scores);
                                ftlog::info!(
                                    "Finished Epoch {}/{num_epochs}: Data {}/{N}, metric {}/{M}, mean roc-auc: {mean_roc_score:.6}",
                                    e + 1,
                                    i + 1,
                                    j + 1,
                                );
                            } else {
                                ftlog::error!("Failed Epoch {}/{num_epochs}, data {}/{N}, metric {}/{M}", e + 1, i + 1, j + 1);
                            }
                        })
                        .map(|(_, v)| v)
                        .collect::<Result<Vec<_>, _>>()
                })
                .last()
                .unwrap_or_else(|| unreachable!("Could not train the model."))?;

            let roc_scores = trained_combinations
                .iter()
                .flat_map(|combos| combos.iter().map(TrainedCombination::expected_roc_score))
                .collect::<Vec<_>>();
            let mean_roc_score: f32 = crate::utils::mean(&roc_scores);
            ftlog::info!(
                "Finished Training epoch {}/{num_epochs} with mean roc-auc: {mean_roc_score}",
                e + 1
            );
        }

        let combinations: [_; M] = trained_combinations.try_into().unwrap_or_else(|v: Vec<_>| {
            unreachable!(
                "Could not convert Vec<Vec<TrainableCombination>> to [Vec<TrainableCombination>; {M}], len: {}",
                v.len()
            )
        });

        for (i, (a, b)) in combinations.iter().zip(self.combinations.iter()).enumerate() {
            if a.len() != b.len() {
                return Err(format!("Mismatch in number of trained combinations for metric {i}"));
            }
        }

        Ok(Chaoda::new(self.metrics.clone(), combinations))
    }
}
