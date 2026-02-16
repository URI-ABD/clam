//! A trainable Single-Metric-CHAODA ensemble.

use rayon::prelude::*;

use crate::{
    chaoda::{inference::TrainedCombination, Graph, OddBall, ParVertex, TrainedSmc, Vertex},
    utils, Dataset, DistanceValue, ParDataset, ParPartition, Partition,
};

use super::{GraphAlgorithm, TrainableCombination, TrainableMetaMlModel};

/// A trainable Single-Metric-CHAODA ensemble.
pub struct TrainableSmc(Vec<TrainableCombination>);

impl TrainableSmc {
    /// Create a new trainable Single-Metric-CHAODA ensemble.
    ///
    /// # Arguments
    ///
    /// - `meta_ml_models`: The `MetaMLModel`s to train with.
    /// - `graph_algorithms`: The `GraphAlgorithm`s to train with.
    ///
    /// # Returns
    ///
    /// The trainable Single-Metric-CHAODA ensemble using all pairs of the given
    /// `MetaMLModel`s and `GraphAlgorithm`s.
    #[must_use]
    pub fn new(meta_ml_models: &[TrainableMetaMlModel], graph_algorithms: &[GraphAlgorithm]) -> Self {
        Self(
            meta_ml_models
                .iter()
                .flat_map(|meta_ml_model| {
                    graph_algorithms.iter().map(move |graph_algorithm| {
                        TrainableCombination::new(meta_ml_model.clone(), graph_algorithm.clone())
                    })
                })
                .collect(),
        )
    }

    /// Get the combinations of `MetaMLModel`s and `GraphAlgorithm`s to train
    /// with.
    #[must_use]
    pub const fn combinations(&self) -> &Vec<TrainableCombination> {
        &self.0
    }

    /// Create trees for use in training.
    ///
    /// # Arguments
    ///
    /// - `datasets`: The datasets to train with.
    /// - `criteria`: The criteria to use for building a tree.
    /// - `metric`: The metric to use for all datasets.
    /// - `seed`: The seed to use for random number generation.
    ///
    /// # Returns
    ///
    /// An array of root vertices for the trees created for each dataset.
    ///
    /// # Type Parameters
    ///
    /// - `N`: The number of datasets, criteria, and root vertices.
    /// - `I`: The type of the items in the datasets.
    /// - `T`: The type of the distance values.
    /// - `D`: The type of the datasets.
    /// - `M`: The type of the metric.
    /// - `S`: The type of the source cluster which will be added to `OddBall`.
    /// - `C`: The type of the criteria function for building a tree.
    pub fn create_trees<const N: usize, I, T, D, M, S, C>(
        &self,
        datasets: &[D; N],
        criteria: &[C; N],
        metric: &M,
    ) -> [OddBall<T, S>; N]
    where
        T: DistanceValue,
        D: Dataset<I>,
        M: Fn(&I, &I) -> T,
        S: Partition<T>,
        C: Fn(&S) -> bool,
    {
        let trees = datasets
            .iter()
            .enumerate()
            .zip(criteria.iter())
            .inspect(|((i, _), _)| ftlog::info!("Creating tree for dataset {}/{N}", i + 1))
            .map(|((_, data), c)| {
                let root = S::new_tree(data, metric, c);
                OddBall::from_cluster_tree(root)
            })
            .collect::<Vec<_>>();
        trees
            .try_into()
            .unwrap_or_else(|e: Vec<_>| unreachable!("Expected {N} trees. Got {} instead", e.len()))
    }

    /// Create graphs to use in the first epoch of training.
    ///
    /// These will be made of vertices with uniform depth from the trees.
    ///
    /// # Arguments
    ///
    /// - `datasets`: The datasets to train with.
    /// - `metric`: The metric to use for all datasets.
    /// - `roots`: The root vertices of the trees.
    /// - `depths`: The depths of the vertices to create.
    ///
    /// # Returns
    ///
    /// An array of graphs (one at each depth) created for each dataset.
    fn create_flat_graphs<'a, const N: usize, I, T, D, M, V>(
        datasets: &[D; N],
        metric: &M,
        roots: &'a [V; N],
        depths: &[usize],
        min_depth: usize,
    ) -> [Vec<Graph<'a, T, V>>; N]
    where
        T: DistanceValue + 'a,
        D: Dataset<I>,
        M: Fn(&I, &I) -> T,
        V: Vertex<T>,
    {
        let graphs = datasets
            .iter()
            .enumerate()
            .zip(roots.iter())
            .inspect(|((i, _), _)| ftlog::info!("Creating flat graphs for dataset {}/{N}", i + 1))
            .map(|((_, data), root)| {
                depths
                    .iter()
                    .enumerate()
                    .inspect(|(j, d)| ftlog::info!("Creating flat graph at depth={d} {}/{}", j + 1, depths.len()))
                    .map(|(_, &depth)| Graph::from_root_uniform_depth(root, data, metric, depth, min_depth))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        graphs
            .try_into()
            .unwrap_or_else(|e: Vec<_>| unreachable!("Expected {N} graphs. Got {} instead", e.len()))
    }

    /// Create graphs for use in training after the first epoch.
    ///
    /// # Arguments
    ///
    /// - `datasets`: The datasets to train with.
    /// - `metric`: The metric to use for all datasets.
    /// - `roots`: The root vertices of the trees.
    /// - `trained_models`: The trained model combinations to use for selecting
    ///   the best clusters.
    /// - `min_depth`: The minimum depth to create graphs for.
    fn create_graphs<'a, const N: usize, I, T, D, M, V>(
        datasets: &[D; N],
        metric: &M,
        roots: &'a [V; N],
        trained_combinations: &[TrainedCombination],
        min_depth: usize,
    ) -> [Vec<Graph<'a, T, V>>; N]
    where
        T: DistanceValue + 'a,
        D: Dataset<I>,
        M: Fn(&I, &I) -> T,
        V: Vertex<T>,
    {
        let graphs = datasets
            .iter()
            .enumerate()
            .zip(roots.iter())
            .inspect(|((i, _), _)| ftlog::info!("Creating meta-ml graphs for dataset {}/{N}", i + 1))
            .map(|((_, data), root)| {
                trained_combinations
                    .iter()
                    .enumerate()
                    .inspect(|(j, combination)| {
                        ftlog::info!(
                            "Creating meta-ml graph {}/{} with {}",
                            j + 1,
                            trained_combinations.len(),
                            combination.name()
                        );
                    })
                    .map(|(_, combination)| combination.create_graph(root, data, metric, min_depth))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        graphs
            .try_into()
            .unwrap_or_else(|e: Vec<_>| unreachable!("Expected {N} graphs. Got {} instead", e.len()))
    }

    /// Train the ensemble for a single epoch.
    ///
    /// # Arguments
    ///
    /// - `graphs`: The graphs to train with.
    /// - `labels`: The labels for the graphs.
    ///
    /// # Returns
    ///
    /// The trained model combinations and the mean roc-auc score.
    ///
    /// # Type Parameters
    ///
    /// - `N`: The number of datasets.
    /// - `T`: The type of the distance values.
    /// - `V`: The type of the vertices in the graphs.
    ///
    /// # Errors
    ///
    /// - If the meta-ml model fails to train.
    fn train_epoch<const N: usize, T: DistanceValue, V: Vertex<T>>(
        &mut self,
        graphs: &[Vec<Graph<T, V>>; N],
        labels: &[Vec<bool>; N],
    ) -> Result<(Vec<TrainedCombination>, f64), String> {
        let num_combinations = self.0.len();

        let (trained_combinations, roc_scores) = self
            .0
            .iter_mut()
            .enumerate()
            .map(|(i, combination)| {
                ftlog::info!(
                    "Training combination {}/{num_combinations} {}",
                    i + 1,
                    combination.name()
                );

                let (x, y, roc_scores_inner) = graphs.iter().enumerate().zip(labels.iter()).fold(
                    (Vec::new(), Vec::new(), Vec::new()),
                    |(mut x, mut y, mut roc_scores_inner), ((j, graphs_inner), labels_inner)| {
                        ftlog::info!("Training dataset {}/{N} with combination {}", j + 1, combination.name());

                        graphs_inner.iter().enumerate().for_each(|(k, graph)| {
                            ftlog::info!(
                                "Training graph {}/{} with combination {}",
                                k + 1,
                                graphs_inner.len(),
                                combination.name()
                            );
                            let ([x_, y_], r) = combination
                                .data_from_graph(graph, labels_inner)
                                .unwrap_or_else(|e| unreachable!("{e}"));
                            x.extend(x_);
                            y.extend(y_);
                            roc_scores_inner.push(r);
                        });

                        (x, y, roc_scores_inner)
                    },
                );

                ftlog::info!(
                    "Appending data for combination {}/{num_combinations} {}",
                    i + 1,
                    combination.name()
                );
                let roc_score = utils::mean(&roc_scores_inner);
                combination
                    .append_data(&x, &y, Some(roc_score))
                    .unwrap_or_else(|e| unreachable!("{e}"));

                ftlog::info!(
                    "Taking training step for combination {}/{num_combinations} {}",
                    i + 1,
                    combination.name()
                );

                combination.train_step::<T, V>().map(|trained| (trained, roc_score))
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let roc_score = utils::mean(&roc_scores);
        Ok((trained_combinations, roc_score))
    }

    /// Train the ensemble.
    ///
    /// # Arguments
    ///
    /// - `datasets`: The datasets to train with.
    /// - `metric`: The metric to use for all datasets.
    /// - `roots`: The root vertices of the trees to use for training.
    /// - `labels`: The labels for the datasets.
    /// - `min_depth`: The minimum depth of clusters to consider for graphs.
    /// - `depths`: The depths to create graphs for the first epoch.
    /// - `num_epochs`: The number of epochs to train for.
    ///
    /// # Returns
    ///
    /// The trained ensemble.
    ///
    /// # Type Parameters
    ///
    /// - `N`: The number of datasets.
    /// - `I`: The type of the items in the datasets.
    /// - `T`: The type of the distance values.
    /// - `D`: The type of the datasets.
    /// - `M`: The type of the metric.
    /// - `V`: The type of the vertices in the graphs.
    ///
    /// # Errors
    ///
    /// - If the minimum depth is less than 1.
    /// - If the number of number if labels is not equal to the cardinality of
    ///   the corresponding dataset.
    /// - If and depth in `depths` is less than `min_depth`.
    /// - If any meta-ml model fails to train.
    #[allow(clippy::too_many_arguments)]
    pub fn train<const N: usize, I, T, D, M, V>(
        &mut self,
        datasets: &[D; N],
        metric: &M,
        roots: &[V; N],
        labels: &[Vec<bool>; N],
        min_depth: usize,
        depths: &[usize],
        num_epochs: usize,
    ) -> Result<TrainedSmc, String>
    where
        T: DistanceValue,
        D: Dataset<I>,
        M: Fn(&I, &I) -> T,
        V: Vertex<T>,
    {
        if min_depth == 0 {
            return Err("Minimum depth must be greater than 0".to_string());
        }

        for (data, labels) in datasets.iter().zip(labels.iter()) {
            if data.cardinality() != labels.len() {
                return Err("Number of data points and labels must be equal".to_string());
            }
        }

        for &d in depths {
            if d < min_depth {
                return Err("Depths must be no smaller than the minimum depth".to_string());
            }
        }

        let flat_graphs = Self::create_flat_graphs(datasets, metric, roots, depths, min_depth);
        let (mut trained_combinations, _) = self.train_epoch(&flat_graphs, labels)?;

        for _ in 1..num_epochs {
            let graphs = Self::create_graphs(datasets, metric, roots, &trained_combinations, min_depth);
            (trained_combinations, _) = self.train_epoch(&graphs, labels)?;
        }

        Ok(TrainedSmc::new(trained_combinations))
    }

    /// Parallel version of [`TrainableSmc::create_trees`](crate::chaoda::training::TrainableSmc::create_trees).
    pub fn par_create_trees<const N: usize, I, T, D, M, S, C>(
        &self,
        datasets: &[D; N],
        criteria: &[C; N],
        metric: &M,
    ) -> [OddBall<T, S>; N]
    where
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        S: ParPartition<T>,
        C: (Fn(&S) -> bool) + Send + Sync,
    {
        let trees = datasets
            .par_iter()
            .enumerate()
            .zip(criteria.par_iter())
            .inspect(|((i, _), _)| ftlog::info!("Creating tree for dataset {}/{N}", i + 1))
            .map(|((_, data), c)| {
                let root = S::par_new_tree(data, metric, c);
                OddBall::from_cluster_tree(root)
            })
            .collect::<Vec<_>>();
        trees
            .try_into()
            .unwrap_or_else(|e: Vec<_>| unreachable!("Expected {N} trees. Got {} instead", e.len()))
    }

    /// Parallel version of [`TrainableSmc::create_flat_graphs`](crate::chaoda::training::TrainableSmc::create_flat_graphs).
    fn par_create_flat_graphs<'a, const N: usize, I, T, D, M, V>(
        datasets: &[D; N],
        metric: &M,
        roots: &'a [V; N],
        depths: &[usize],
        min_depth: usize,
    ) -> [Vec<Graph<'a, T, V>>; N]
    where
        I: Send + Sync,
        T: DistanceValue + Send + Sync + 'a,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        V: ParVertex<T>,
    {
        let graphs = datasets
            .par_iter()
            .enumerate()
            .zip(roots.par_iter())
            .inspect(|((i, _), _)| ftlog::info!("Creating flat graphs for dataset {}/{N}", i + 1))
            .map(|((_, data), root)| {
                depths
                    .par_iter()
                    .enumerate()
                    .inspect(|(j, d)| ftlog::info!("Creating flat graph at depth={d} {}/{}", j + 1, depths.len()))
                    .map(|(_, &depth)| Graph::par_from_root_uniform_depth(root, data, metric, depth, min_depth))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        graphs
            .try_into()
            .unwrap_or_else(|e: Vec<_>| unreachable!("Expected {N} graphs. Got {} instead", e.len()))
    }

    /// Parallel version of [`TrainableSmc::create_graphs`](crate::chaoda::training::TrainableSmc::create_graphs).
    fn par_create_graphs<'a, const N: usize, I, T, D, M, V>(
        datasets: &[D; N],
        metric: &M,
        roots: &'a [V; N],
        trained_combinations: &[TrainedCombination],
        min_depth: usize,
    ) -> [Vec<Graph<'a, T, V>>; N]
    where
        I: Send + Sync,
        T: DistanceValue + Send + Sync + 'a,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        V: ParVertex<T>,
    {
        let graphs = datasets
            .par_iter()
            .enumerate()
            .zip(roots.par_iter())
            .inspect(|((i, _), _)| ftlog::info!("Creating meta-ml graphs for dataset {}/{N}", i + 1))
            .map(|((_, data), root)| {
                trained_combinations
                    .par_iter()
                    .enumerate()
                    .inspect(|(j, combination)| {
                        ftlog::info!(
                            "Creating meta-ml graph {}/{} with {}",
                            j + 1,
                            trained_combinations.len(),
                            combination.name()
                        );
                    })
                    .map(|(_, combination)| combination.par_create_graph(root, data, metric, min_depth))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        graphs
            .try_into()
            .unwrap_or_else(|e: Vec<_>| unreachable!("Expected {N} graphs. Got {} instead", e.len()))
    }

    /// Parallel version of [`TrainableSmc::train_epoch`](crate::chaoda::training::TrainableSmc::train_epoch).
    ///
    /// # Errors
    ///
    /// - See [`TrainableSmc::train_epoch`](crate::chaoda::training::TrainableSmc::train_epoch).
    fn par_train_epoch<const N: usize, T: DistanceValue + Send + Sync, V: ParVertex<T>>(
        &mut self,
        graphs: &[Vec<Graph<T, V>>; N],
        labels: &[Vec<bool>; N],
    ) -> Result<(Vec<TrainedCombination>, f64), String> {
        let num_combinations = self.0.len();

        let (trained_combinations, roc_scores) = self
            .0
            .par_iter_mut()
            .enumerate()
            .map(|(i, combination)| {
                ftlog::info!(
                    "Training combination {}/{num_combinations} {}",
                    i + 1,
                    combination.name()
                );

                let (x, y, roc_scores_inner) = graphs
                    .par_iter()
                    .enumerate()
                    .zip(labels.par_iter())
                    .fold(
                        || (Vec::new(), Vec::new(), Vec::new()),
                        |(mut x, mut y, mut roc_scores_inner), ((j, graphs_inner), labels_inner)| {
                            ftlog::info!("Training dataset {}/{N} with combination {}", j + 1, combination.name());

                            graphs_inner.iter().enumerate().for_each(|(k, graph)| {
                                ftlog::info!(
                                    "Training graph {}/{} with combination {}",
                                    k + 1,
                                    graphs_inner.len(),
                                    combination.name()
                                );
                                let ([x_, y_], r) = combination
                                    .data_from_graph(graph, labels_inner)
                                    .unwrap_or_else(|e| unreachable!("{e}"));
                                x.extend(x_);
                                y.extend(y_);
                                roc_scores_inner.push(r);
                            });

                            (x, y, roc_scores_inner)
                        },
                    )
                    .reduce(
                        || (Vec::new(), Vec::new(), Vec::new()),
                        |(mut x1, mut y1, mut roc_scores_inner1), (x2, y2, roc_scores_inner2)| {
                            x1.extend(x2);
                            y1.extend(y2);
                            roc_scores_inner1.extend(roc_scores_inner2);
                            (x1, y1, roc_scores_inner1)
                        },
                    );

                ftlog::info!(
                    "Appending data for combination {}/{num_combinations} {}",
                    i + 1,
                    combination.name()
                );
                let roc_score = utils::mean(&roc_scores_inner);
                combination
                    .append_data(&x, &y, Some(roc_score))
                    .unwrap_or_else(|e| unreachable!("{e}"));

                ftlog::info!(
                    "Taking training step for combination {}/{num_combinations} {}",
                    i + 1,
                    combination.name()
                );

                combination.train_step::<T, V>().map(|trained| (trained, roc_score))
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let roc_score = utils::mean(&roc_scores);
        Ok((trained_combinations, roc_score))
    }

    /// Parallel version of [`TrainableSmc::train`](crate::chaoda::training::TrainableSmc::train).
    ///
    /// # Errors
    ///
    /// - See [`TrainableSmc::train`](crate::chaoda::training::TrainableSmc::train).
    #[allow(clippy::too_many_arguments)]
    pub fn par_train<const N: usize, I, T, D, M, V>(
        &mut self,
        datasets: &[D; N],
        metric: &M,
        roots: &[V; N],
        labels: &[Vec<bool>; N],
        min_depth: usize,
        depths: &[usize],
        num_epochs: usize,
    ) -> Result<TrainedSmc, String>
    where
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        V: ParVertex<T>,
    {
        if min_depth == 0 {
            return Err("Minimum depth must be greater than 0".to_string());
        }

        for (data, labels) in datasets.iter().zip(labels.iter()) {
            if data.cardinality() != labels.len() {
                return Err("Number of data points and labels must be equal".to_string());
            }
        }

        for &d in depths {
            if d < min_depth {
                return Err("Depths must be no smaller than the minimum depth".to_string());
            }
        }

        let flat_graphs = Self::par_create_flat_graphs(datasets, metric, roots, depths, min_depth);
        let (mut trained_combinations, _) = self.par_train_epoch(&flat_graphs, labels)?;

        for _ in 1..num_epochs {
            let graphs = Self::par_create_graphs(datasets, metric, roots, &trained_combinations, min_depth);
            (trained_combinations, _) = self.par_train_epoch(&graphs, labels)?;
        }

        Ok(TrainedSmc::new(trained_combinations))
    }
}
