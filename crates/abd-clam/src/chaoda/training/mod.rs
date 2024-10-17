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

use super::{inference::Chaoda, Graph, Vertex};

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
                metric_trees.push(Vertex::adapt_tree(root, None, data));
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

    /// Create graphs for use in training the first epoch.
    #[allow(clippy::type_complexity)]
    fn create_flat_graphs<'a, const N: usize, D: Dataset<I, U>, S: Cluster<I, U, D>>(
        &self,
        datasets: &mut [D; N],
        trees: &'a [[Vertex<I, U, D, S>; M]; N],
        depths: &[usize],
    ) -> [[Vec<Graph<'a, I, U, D, S>>; M]; N] {
        let mut graphs_vmn = Vec::new();
        ftlog::info!("Creating flat graphs...");

        for (i, (data, roots)) in datasets.iter_mut().zip(trees.iter()).enumerate() {
            let mut graphs_vm = Vec::new();
            ftlog::info!("Creating flat graphs for dataset {}/{N}...", i + 1);

            for (j, (metric, root)) in self.metrics.iter().zip(roots.iter()).enumerate() {
                data.set_metric(metric.clone());
                ftlog::info!(
                    "Creating flat graphs for dataset {}/{N}, metric {}/{M}...",
                    i + 1,
                    j + 1
                );

                let mut graphs_v = Vec::new();
                for &depth in depths {
                    ftlog::info!(
                        "Creating flat graphs for dataset {}/{N}, metric {}/{M}, depth {depth}/{}",
                        i + 1,
                        j + 1,
                        depths.len()
                    );
                    let cluster_scorer = |clusters: &[&Vertex<I, U, D, S>]| {
                        clusters
                            .iter()
                            .map(|c| {
                                if c.depth() == depth || (c.is_leaf() && c.depth() < depth) {
                                    1.0
                                } else {
                                    0.0
                                }
                            })
                            .collect::<Vec<_>>()
                    };
                    graphs_v.push(Graph::from_root(root, data, cluster_scorer, depth));
                    ftlog::info!(
                        "Finished flat graphs for dataset {}/{N}, metric {}/{M}, depth {depth}/{}",
                        i + 1,
                        j + 1,
                        depths.len()
                    );
                }

                ftlog::info!(
                    "Finished creating flat graphs for dataset {}/{N}, metric {}/{M}",
                    i + 1,
                    j + 1
                );
                graphs_vm.push(graphs_v);
            }

            let graphs_vm = graphs_vm.try_into().unwrap_or_else(|v: Vec<_>| {
                unreachable!(
                    "Could not convert Vec<Vec<Graph>> to [Vec<Graph>; {M}]. Len was {}",
                    v.len()
                )
            });
            ftlog::info!("Finished creating flat graphs for dataset {}/{N}", i + 1);

            graphs_vmn.push(graphs_vm);
        }

        let graphs_vmn = graphs_vmn.try_into().unwrap_or_else(|v: Vec<_>| {
            unreachable!(
                "Could not convert Vec<[Vec<Graph>; {M}]> to [[Vec<Graph>; {M}]; {N}]. Len was {}",
                v.len()
            )
        });
        ftlog::info!("Finished creating flat graphs");

        graphs_vmn
    }

    /// Create graphs for use in training.
    ///
    /// # Arguments
    ///
    /// - `datasets`: The datasets to create graphs for.
    /// - `trees`: The trees to use for creating the graphs.
    /// - `min_depth`: The minimum depth of `Cluster`s to use for `Graph`s.
    ///
    /// # Returns
    ///
    /// The graphs created from the datasets. There will be a `Graph` for each
    /// combination of `Dataset`, `Metric`, and pair of `MetaMLModel` and
    /// `GraphAlgorithm`.
    #[allow(clippy::type_complexity)]
    fn create_graphs<'a, const N: usize, D: Dataset<I, U>, S: Cluster<I, U, D>>(
        &self,
        datasets: &mut [D; N],
        trees: &'a [[Vertex<I, U, D, S>; M]; N],
        trained_models: &[Vec<TrainedCombination>; M],
        min_depth: usize,
    ) -> [[Vec<Graph<'a, I, U, D, S>>; M]; N] {
        let mut graphs_vmn = Vec::new();
        ftlog::info!("Creating graphs...");

        for (i, (data, trees)) in datasets.iter_mut().zip(trees.iter()).enumerate() {
            let mut graphs_vm = Vec::new();
            ftlog::info!("Creating graphs for dataset {}/{N}...", i + 1);

            for (j, ((metric, root), trained_models)) in self
                .metrics
                .iter()
                .zip(trees.iter())
                .zip(trained_models.iter())
                .enumerate()
            {
                data.set_metric(metric.clone());
                ftlog::info!("Creating graphs for dataset {}/{N}, metric {}/{M}...", i + 1, j + 1);
                graphs_vm.push(
                    trained_models
                        .iter()
                        .enumerate()
                        .inspect(|(k, _)| {
                            ftlog::info!(
                                "Creating graph for dataset {}/{N}, metric {}/{M}, model {}/{M}...",
                                i + 1,
                                j + 1,
                                k + 1
                            );
                        })
                        .map(|(k, combination)| (k, combination.create_graph(root, data, min_depth)))
                        .inspect(|(k, _)| {
                            ftlog::info!(
                                "Finished graph for dataset {}/{N}, metric {}/{M}, model {}/{M}...",
                                i + 1,
                                j + 1,
                                k + 1
                            );
                        })
                        .map(|(_, combination)| combination)
                        .collect::<Vec<_>>(),
                );
                ftlog::info!(
                    "Finished creating graphs for dataset {}/{N}, metric {}/{M}",
                    i + 1,
                    j + 1
                );
            }

            let graphs_vm = graphs_vm.try_into().unwrap_or_else(|v: Vec<_>| {
                unreachable!("Could not convert Vec<Graph> to [Vec<Graph>; {M}]. Len was {}", v.len())
            });
            ftlog::info!("Finished creating graphs for dataset {}/{N}", i + 1);
            graphs_vmn.push(graphs_vm);
        }

        let graphs_vmn = graphs_vmn.try_into().unwrap_or_else(|v: Vec<_>| {
            unreachable!(
                "Could not convert Vec<[Vec<Graph>; {M}]> to [[Vec<Graph>; {M}]; {N}]. Len was {}",
                v.len()
            )
        });
        ftlog::info!("Finished creating graphs");
        graphs_vmn
    }

    /// Train the model for an epoch.
    ///
    /// # Arguments
    ///
    /// - `graphs`: The graphs to train with.
    /// - `labels`: The labels for the graphs.
    ///
    /// # Returns
    ///
    /// The trained combinations and the mean roc-auc score.
    #[allow(clippy::type_complexity)]
    fn train_epoch<const N: usize, D: Dataset<I, U>, S: Cluster<I, U, D>>(
        &mut self,
        graphs: &[[Vec<Graph<I, U, D, S>>; M]; N],
        labels: &[Vec<bool>; N],
    ) -> Result<([Vec<TrainedCombination>; M], f32), String> {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut roc_scores = Vec::new();

        for ((graphs, labels), combinations) in graphs.iter().zip(labels.iter()).zip(self.combinations.iter()) {
            for (graphs, combination) in graphs.iter().zip(combinations.iter()) {
                for graph in graphs {
                    let ([mut x_, mut y_], roc_score) = combination.data_from_graph(graph, labels)?;
                    x.append(&mut x_);
                    y.append(&mut y_);
                    roc_scores.push(roc_score);
                }
            }
        }

        let roc_score: f32 = crate::utils::mean(&roc_scores);

        let mut combinations_vm = Vec::new();

        for combinations in &mut self.combinations {
            let mut combinations_v = Vec::new();
            for combination in combinations {
                combination.append_data(&x, &y, Some(roc_score))?;
                let trained_combination = combination.train_step()?;
                combinations_v.push(trained_combination);
            }
            combinations_vm.push(combinations_v);
        }

        let trained_combinations = combinations_vm.try_into().unwrap_or_else(|v: Vec<_>| {
            unreachable!(
                "Could not convert Vec<Vec<TrainableCombination>> to [Vec<TrainedCombination>; {M}], len: {}",
                v.len()
            )
        });

        Ok((trained_combinations, roc_score))
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
    /// - If the minimum depth is 0.
    /// - If the number of epochs is 0.
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
        if min_depth == 0 {
            return Err("Minimum depth must be greater than 0.".to_string());
        }

        if num_epochs == 0 {
            return Err("Number of epochs must be greater than 0.".to_string());
        }

        for (data, labels) in datasets.iter().zip(labels.iter()) {
            if data.cardinality() != labels.len() {
                return Err("The number of labels does not match the number of data points.".to_string());
            }
        }
        ftlog::info!("Training with {N} datasets and {M} metrics...");

        // For the first epoch, we train with Graphs at various uniform depths
        // in the trees. The depths are multiples of the minimum depth.
        let flat_depths = (min_depth..)
            .step_by(min_depth)
            .take(self.combinations[0].len())
            .collect::<Vec<_>>();
        let flat_graphs = self.create_flat_graphs(datasets, trees, &flat_depths);
        let (mut trained_combinations, mut roc_score) = self.train_epoch(&flat_graphs, labels)?;
        ftlog::info!("Finished Training epoch 1/{num_epochs} with mean roc-auc: {roc_score}");

        for e in 1..num_epochs {
            // For the remaining epochs, we train with Graphs made using trained meta-ml models
            let graphs = self.create_graphs(datasets, trees, &trained_combinations, min_depth);
            (trained_combinations, roc_score) = self.train_epoch(&graphs, labels)?;
            ftlog::info!(
                "Finished Training epoch {}/{num_epochs} with mean roc-auc: {roc_score}",
                e + 1
            );
        }

        Ok(Chaoda::new(self.metrics.clone(), trained_combinations))
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
                metric_trees.push(Vertex::par_adapt_tree(root, None, data));
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

    /// Parallel version of `create_flat_graphs`.
    #[allow(clippy::type_complexity)]
    fn par_create_flat_graphs<'a, const N: usize, D: ParDataset<I, U>, S: ParCluster<I, U, D>>(
        &self,
        datasets: &mut [D; N],
        trees: &'a [[Vertex<I, U, D, S>; M]; N],
        depths: &[usize],
    ) -> [[Vec<Graph<'a, I, U, D, S>>; M]; N] {
        let mut graphs_vmn = Vec::new();
        ftlog::info!("Creating flat graphs...");

        for (i, (data, roots)) in datasets.iter_mut().zip(trees.iter()).enumerate() {
            let mut graphs_vm = Vec::new();
            ftlog::info!("Creating flat graphs for dataset {}/{N}...", i + 1);

            for (j, (metric, root)) in self.metrics.iter().zip(roots.iter()).enumerate() {
                data.set_metric(metric.clone());
                ftlog::info!(
                    "Creating flat graphs for dataset {}/{N}, metric {}/{M}...",
                    i + 1,
                    j + 1
                );

                graphs_vm.push(
                    depths
                        .par_iter()
                        .inspect(|&&depth| {
                            ftlog::info!(
                                "Creating flat graph for dataset {}/{N}, metric {}/{M}, depth {depth}/{}",
                                i + 1,
                                j + 1,
                                depths.len()
                            );
                        })
                        .map(|&depth| {
                            let cluster_scorer = |clusters: &[&Vertex<I, U, D, S>]| {
                                clusters
                                    .iter()
                                    .map(|c| {
                                        if c.depth() == depth || (c.is_leaf() && c.depth() < depth) {
                                            1.0
                                        } else {
                                            0.0
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            };
                            let graph = Graph::par_from_root(root, data, cluster_scorer, depth);
                            (depth, graph)
                        })
                        .inspect(|&(depth, _)| {
                            ftlog::info!(
                                "Finished flat graph for dataset {}/{N}, metric {}/{M}, depth {depth}/{}",
                                i + 1,
                                j + 1,
                                depths.len()
                            );
                        })
                        .map(|(_, graph)| graph)
                        .collect::<Vec<_>>(),
                );

                ftlog::info!(
                    "Finished creating flat graphs for dataset {}/{N}, metric {}/{M}",
                    i + 1,
                    j + 1
                );
            }

            let graphs_vm = graphs_vm.try_into().unwrap_or_else(|v: Vec<_>| {
                unreachable!(
                    "Could not convert Vec<Vec<Graph>> to [Vec<Graph>; {M}]. Len was {}",
                    v.len()
                )
            });
            ftlog::info!("Finished creating flat graphs for dataset {}/{N}", i + 1);
            graphs_vmn.push(graphs_vm);
        }

        let graphs_vmn = graphs_vmn.try_into().unwrap_or_else(|v: Vec<_>| {
            unreachable!(
                "Could not convert Vec<[Vec<Graph>; {M}]> to [[Vec<Graph>; {M}]; {N}]. Len was {}",
                v.len()
            )
        });

        ftlog::info!("Finished creating flat graphs");
        graphs_vmn
    }

    /// Parallel version of `create_graphs`.
    #[allow(clippy::type_complexity)]
    fn par_create_graphs<'a, const N: usize, D: ParDataset<I, U>, S: ParCluster<I, U, D>>(
        &self,
        datasets: &mut [D; N],
        trees: &'a [[Vertex<I, U, D, S>; M]; N],
        trained_models: &[Vec<TrainedCombination>; M],
        min_depth: usize,
    ) -> [[Vec<Graph<'a, I, U, D, S>>; M]; N] {
        let mut graphs_vmn = Vec::new();
        ftlog::info!("Creating graphs...");

        for (i, (data, trees)) in datasets.iter_mut().zip(trees.iter()).enumerate() {
            let mut graphs_vm = Vec::new();
            ftlog::info!("Creating graphs for dataset {}/{N}...", i + 1);

            for (j, ((metric, root), trained_models)) in self
                .metrics
                .iter()
                .zip(trees.iter())
                .zip(trained_models.iter())
                .enumerate()
            {
                data.set_metric(metric.clone());
                ftlog::info!("Creating graphs for dataset {}/{N}, metric {}/{M}...", i + 1, j + 1);
                graphs_vm.push(
                    trained_models
                        .par_iter()
                        .enumerate()
                        .inspect(|(k, _)| {
                            ftlog::info!(
                                "Creating graph for dataset {}/{N}, metric {}/{M}, model {}/{M}...",
                                i + 1,
                                j + 1,
                                k + 1
                            );
                        })
                        .map(|(k, combination)| (k, combination.par_create_graph(root, data, min_depth)))
                        .inspect(|(k, _)| {
                            ftlog::info!(
                                "Finished graph for dataset {}/{N}, metric {}/{M}, model {}/{M}...",
                                i + 1,
                                j + 1,
                                k + 1
                            );
                        })
                        .map(|(_, combination)| combination)
                        .collect::<Vec<_>>(),
                );
                ftlog::info!(
                    "Finished creating graphs for dataset {}/{N}, metric {}/{M}",
                    i + 1,
                    j + 1
                );
            }

            let graphs_vm = graphs_vm.try_into().unwrap_or_else(|v: Vec<_>| {
                unreachable!("Could not convert Vec<Graph> to [Vec<Graph>; {M}]. Len was {}", v.len())
            });
            ftlog::info!("Finished creating graphs for dataset {}/{N}", i + 1);
            graphs_vmn.push(graphs_vm);
        }

        let graphs_vmn = graphs_vmn.try_into().unwrap_or_else(|v: Vec<_>| {
            unreachable!(
                "Could not convert Vec<[Vec<Graph>; {M}]> to [[Vec<Graph>; {M}]; {N}]. Len was {}",
                v.len()
            )
        });
        ftlog::info!("Finished creating graphs");
        graphs_vmn
    }

    /// Parallel version of `train_epoch`.
    #[allow(clippy::type_complexity)]
    fn par_train_epoch<const N: usize, D: ParDataset<I, U>, S: ParCluster<I, U, D>>(
        &mut self,
        graphs: &[[Vec<Graph<I, U, D, S>>; M]; N],
        labels: &[Vec<bool>; N],
    ) -> Result<([Vec<TrainedCombination>; M], f32), String> {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut roc_scores = Vec::new();

        ftlog::info!("Extracting data from graphs...");
        for ((graphs, labels), combinations) in graphs.iter().zip(labels.iter()).zip(self.combinations.iter()) {
            for (graphs, combination) in graphs.iter().zip(combinations.iter()) {
                for graph in graphs {
                    let ([mut x_, mut y_], roc_score) = combination.data_from_graph(graph, labels)?;
                    x.append(&mut x_);
                    y.append(&mut y_);
                    roc_scores.push(roc_score);
                }
            }
        }
        ftlog::info!("Finished extracting data from graphs");
        ftlog::info!("Got {} data points", y.len());

        let roc_score: f32 = crate::utils::mean(&roc_scores);
        ftlog::info!("Mean roc-auc: {roc_score:.6}");

        ftlog::info!("Training models...");
        let trained_combinations = self
            .combinations
            .par_iter_mut()
            .enumerate()
            .inspect(|(i, _)| ftlog::info!("Training models for metric {}/{M}...", i + 1))
            .map(|(i, combinations)| {
                let combinations = combinations
                    .par_iter_mut()
                    .map(|combination| {
                        combination.append_data(&x, &y, Some(roc_score))?;
                        combination.train_step()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok::<_, String>((i, combinations))
            })
            .inspect(|val| {
                if let Ok((i, _)) = val {
                    ftlog::info!("Finished training models for metric {}/{M}", i + 1);
                }
            })
            .map(|val| val.map(|(_, combinations)| combinations))
            .collect::<Result<Vec<_>, _>>()?;

        let trained_combinations = trained_combinations.try_into().unwrap_or_else(|v: Vec<_>| {
            unreachable!(
                "Could not convert Vec<Vec<TrainableCombination>> to [Vec<TrainedCombination>; {M}], len: {}",
                v.len()
            )
        });

        Ok((trained_combinations, roc_score))
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
        if min_depth == 0 {
            return Err("Minimum depth must be greater than 0.".to_string());
        }

        if num_epochs == 0 {
            return Err("Number of epochs must be greater than 0.".to_string());
        }

        for (data, labels) in datasets.iter().zip(labels.iter()) {
            if data.cardinality() != labels.len() {
                return Err("The number of labels does not match the number of data points.".to_string());
            }
        }
        ftlog::info!("Training with {N} datasets and {M} metrics...");

        // For the first epoch, we train with Graphs at various uniform depths
        // in the trees. The depths are multiples of the minimum depth.
        ftlog::info!("Starting training...");
        let flat_depths = (min_depth..)
            .step_by(min_depth)
            .take(self.combinations[0].len())
            .collect::<Vec<_>>();
        let flat_graphs = self.par_create_flat_graphs(datasets, trees, &flat_depths);
        let (mut trained_combinations, mut roc_score) = self.par_train_epoch(&flat_graphs, labels)?;
        ftlog::info!("Finished Training epoch 1/{num_epochs} with mean roc-auc: {roc_score}");

        for e in 1..num_epochs {
            // For the remaining epochs, we train with Graphs made using trained meta-ml models
            ftlog::info!("Starting Training epoch {}/{}...", e + 1, num_epochs);
            let graphs = self.par_create_graphs(datasets, trees, &trained_combinations, min_depth);
            (trained_combinations, roc_score) = self.par_train_epoch(&graphs, labels)?;
            ftlog::info!(
                "Finished Training epoch {}/{num_epochs} with mean roc-auc: {roc_score}",
                e + 1
            );
        }

        Ok(Chaoda::new(self.metrics.clone(), trained_combinations))
    }
}
