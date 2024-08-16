//! A CHAODA model that works with a single metric and tree.

use distances::Number;
use mt_logger::{mt_log, Level};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smartcore::metrics::roc_auc_score;

use crate::{chaoda::members::Algorithm, cluster::ParCluster, dataset::ParDataset, Cluster, Dataset, Metric};

use super::{Graph, Member, MlModel, Vertex};

/// The type of the training data for the meta-ml models.
///
/// - The outer `Vec` contains the training data for each combination of `Member` and `MlModel`.
/// - The inner `Vec` contains the training data for a single `MlModel`.
/// - The tuples contain the properties of the instances and the roc-auc score.
pub type TrainingData = Vec<Vec<(Vec<f32>, f32)>>;

/// The combination of `Member` and `MlModel`, and their corresponding `Graph`s,
/// that are used in the ensemble.
type Models = Vec<(Member, MlModel)>;

/// A CHAODA model that works with a single metric and tree.
#[derive(Serialize, Deserialize)]
pub struct SingleMetricModel<I: Clone, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> {
    /// The metric used to calculate the distance between two instances.
    #[serde(skip_serializing, skip_deserializing)]
    metric: Metric<I, U>,
    /// The combinations of `Member` and `MlModel` that are used in the ensemble.
    models: Models,
    /// The minimum depth in the tree to consider when selecting nodes for the `Graph`s.
    min_depth: usize,
    /// Phantom data to satisfy the compiler.
    _p: std::marker::PhantomData<(D, S)>,
}

impl<I: Clone, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> SingleMetricModel<I, U, D, S> {
    /// Create a new `SingleMetricModel`.
    #[must_use]
    pub fn new(metric: Metric<I, U>, model_combinations: Vec<(Member, Vec<MlModel>)>, min_depth: usize) -> Self {
        let models = model_combinations
            .into_iter()
            .flat_map(|(member, models)| models.into_iter().map(move |model| (member.clone(), model)))
            .collect();
        Self {
            metric,
            models,
            min_depth,
            _p: std::marker::PhantomData,
        }
    }

    /// Changes the metric used to calculate the distance between two instances.
    pub fn with_metric(mut self, metric: Metric<I, U>) -> Self {
        self.metric = metric;
        self
    }

    /// Train the model for one epoch.
    ///
    /// # Parameters
    ///
    /// - `data`: The dataset to use for training.
    /// - `root`: The root of the tree.
    /// - `labels`: The labels for the instances in the dataset.
    /// - `training_data`: The training data from previous epochs.
    ///
    /// # Returns
    ///
    /// The training data after generated during this epoch along with the
    /// training data from previous epochs.
    ///
    /// # Errors
    ///
    /// - If the number of labels does not match the number of instances in the
    ///   dataset.
    /// - If the training of the meta-ml models fails.
    pub fn train_step(
        &mut self,
        data: &mut D,
        root: &Vertex<I, U, D, S>,
        labels: &[bool],
        training_data: &TrainingData,
    ) -> Result<TrainingData, String> {
        if labels.len() != data.cardinality() {
            return Err(format!(
                "The number of labels ({}) does not match the number of instances ({})",
                labels.len(),
                data.cardinality()
            ));
        }

        let y_true = labels.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect::<Vec<_>>();

        let mut graphs = if training_data.is_empty() {
            mt_log!(
                Level::Info,
                "Creating default graphs for {} {}...",
                data.name(),
                data.metric().name()
            );
            self.create_default_graphs(data, root)
        } else {
            mt_log!(
                Level::Info,
                "Creating meta-ml graphs for {} {}...",
                data.name(),
                data.metric().name()
            );
            self.create_graphs(data, root)
        };

        let step_data = self.generate_training_data(&y_true, &mut graphs);
        self.train_models(training_data, &step_data)?;

        mt_log!(
            Level::Info,
            "Training step completed on {} {}. Generated {} samples...",
            data.name(),
            data.metric().name(),
            step_data.iter().map(Vec::len).sum::<usize>()
        );

        Ok(step_data)
    }

    /// Predict the anomaly scores for the instances in the dataset for each
    /// combination of `Member` and `MlModel`.
    pub fn predict(&self, data: &mut D, root: &Vertex<I, U, D, S>) -> Vec<Vec<f32>> {
        data.set_metric(self.metric.clone());

        mt_log!(
            Level::Info,
            "Predicting scores for {} {}...",
            data.name(),
            data.metric().name()
        );
        let mut graphs = self.create_graphs(data, root);

        self.models
            .iter()
            .zip(graphs.iter_mut())
            .map(|((member, _), g)| member.evaluate_points(g))
            .collect()
    }

    /// Creates `Graph`s for each combination of `Member` and `MlModel`.
    ///
    /// This will also change the metric in the `data` so it is upon the user to
    /// change it back if needed.
    fn create_graphs<'a>(&self, data: &mut D, root: &'a Vertex<I, U, D, S>) -> Vec<Graph<'a, I, U, D, S>> {
        data.set_metric(self.metric.clone());

        self.models
            .iter()
            .map(|(member, model)| {
                let cluster_scorer = |vertices: &[&Vertex<I, U, D, S>]| {
                    let properties = vertices.iter().map(|v| v.ratios().to_vec()).collect::<Vec<_>>();
                    model
                        .predict(&properties)
                        .unwrap_or_else(|e| unreachable!("Failed to predict: {e}"))
                };
                let g = Graph::from_tree(root, data, cluster_scorer, self.min_depth);
                mt_log!(
                    Level::Info,
                    "Graph created, via meta-ml, with {} vertices for {} {} under {} {}...",
                    g.cardinality(),
                    data.name(),
                    data.metric().name(),
                    member.name(),
                    model.name()
                );
                g
            })
            .collect()
    }

    /// Creates `Graph`s with vertices at the min-depth.
    fn create_default_graphs<'a>(&self, data: &mut D, root: &'a Vertex<I, U, D, S>) -> Vec<Graph<'a, I, U, D, S>> {
        data.set_metric(self.metric.clone());

        self.models
            .iter()
            .map(|(member, model)| {
                let cluster_scorer = |vertices: &[&Vertex<I, U, D, S>]| {
                    vertices
                        .iter()
                        .map(|v| {
                            if (v.depth() == self.min_depth) || (v.is_leaf() && v.depth() < self.min_depth) {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect::<Vec<_>>()
                };
                let g = Graph::from_tree(root, data, cluster_scorer, self.min_depth);
                mt_log!(
                    Level::Info,
                    "Default Graph created, with {} vertices for {} {} under {} {}...",
                    g.cardinality(),
                    data.name(),
                    data.metric().name(),
                    member.name(),
                    model.name()
                );
                g
            })
            .collect()
    }

    /// Generate the training data to use for the meta-ml models.
    fn generate_training_data(&self, y_true: &[f32], graphs: &mut [Graph<I, U, D, S>]) -> TrainingData {
        self.models
            .iter()
            .zip(graphs.iter_mut())
            .map(|((member, _), graph)| {
                let anomaly_ratings = member.evaluate_points(graph);
                graph
                    .iter_clusters()
                    .map(|v| {
                        let train_x = v.ratios().to_vec();

                        let (y_true, y_pred) = v
                            .indices()
                            .map(|i| (y_true[i], anomaly_ratings[i]))
                            .chain(core::iter::once((0.0, 0.0)))
                            .chain(core::iter::once((1.0, 1.0)))
                            .unzip::<_, _, Vec<_>, Vec<_>>();

                        let roc_auc = roc_auc_score(&y_true, &y_pred).as_f32();

                        (train_x, roc_auc)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Train the meta-ml models.
    fn train_models(&mut self, training_data: &TrainingData, step_data: &TrainingData) -> Result<(), String> {
        mt_log!(Level::Info, "Training models with {} samples...", training_data.len());

        let training_data = training_data
            .iter()
            .zip(step_data.iter())
            .map(|(a, b)| a.iter().chain(b.iter()).cloned().collect::<Vec<_>>());

        self.models
            .iter_mut()
            .zip(training_data)
            .try_for_each(|((_, ml_model), ml_data)| {
                let (train_x, train_y) = ml_data.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                ml_model.train(&train_x, &train_y)
            })
    }
}

impl<I: Clone + Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<I, U, D>> SingleMetricModel<I, U, D, S> {
    /// Parallel version of `train_step`.
    pub fn par_train_step(
        &mut self,
        data: &mut D,
        root: &Vertex<I, U, D, S>,
        labels: &[bool],
        training_data: &TrainingData,
    ) -> Result<TrainingData, String> {
        if labels.len() != data.cardinality() {
            return Err(format!(
                "The number of labels ({}) does not match the number of instances ({})",
                labels.len(),
                data.cardinality()
            ));
        }

        let y_true = labels.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect::<Vec<_>>();

        let mut graphs = if training_data.is_empty() {
            mt_log!(
                Level::Info,
                "Creating default graphs for {} {}...",
                data.name(),
                data.metric().name()
            );
            self.par_create_default_graphs(data, root)
        } else {
            mt_log!(
                Level::Info,
                "Creating meta-ml graphs for {} {}...",
                data.name(),
                data.metric().name()
            );
            self.par_create_graphs(data, root)
        };

        let step_data = self.par_generate_training_data(&y_true, &mut graphs);
        self.par_train_models(training_data, &step_data)?;

        mt_log!(
            Level::Info,
            "Training step completed on {} {}. Generated {} samples...",
            data.name(),
            data.metric().name(),
            step_data.iter().map(Vec::len).sum::<usize>()
        );

        Ok(step_data)
    }

    /// Parallel version of `predict`.
    pub fn par_predict(&self, data: &mut D, root: &Vertex<I, U, D, S>) -> Vec<Vec<f32>> {
        data.set_metric(self.metric.clone());
        let mut graphs = self.par_create_graphs(data, root);

        self.models
            .par_iter()
            .zip(graphs.par_iter_mut())
            .map(|((member, _), g)| member.evaluate_points(g))
            .collect()
    }

    /// Parallel version of `create_graphs`.
    fn par_create_graphs<'a>(&self, data: &mut D, root: &'a Vertex<I, U, D, S>) -> Vec<Graph<'a, I, U, D, S>> {
        data.set_metric(self.metric.clone());

        self.models
            .par_iter()
            .map(|(member, model)| {
                let cluster_scorer = |vertices: &[&Vertex<I, U, D, S>]| {
                    let properties = vertices.iter().map(|v| v.ratios().to_vec()).collect::<Vec<_>>();
                    model
                        .predict(&properties)
                        .unwrap_or_else(|e| unreachable!("Failed to predict: {e}"))
                };
                let g = Graph::from_tree(root, data, cluster_scorer, self.min_depth);
                mt_log!(
                    Level::Info,
                    "Graph created, via meta-ml, with {} vertices for {} {} under {} {}...",
                    g.cardinality(),
                    data.name(),
                    data.metric().name(),
                    member.name(),
                    model.name()
                );
                g
            })
            .collect()
    }

    /// Parallel version of `create_default_graphs`.
    fn par_create_default_graphs<'a>(&self, data: &mut D, root: &'a Vertex<I, U, D, S>) -> Vec<Graph<'a, I, U, D, S>> {
        data.set_metric(self.metric.clone());

        self.models
            .par_iter()
            .map(|(member, model)| {
                let cluster_scorer = |vertices: &[&Vertex<I, U, D, S>]| {
                    vertices
                        .iter()
                        .map(|v| {
                            if (v.depth() == self.min_depth) || (v.is_leaf() && v.depth() < self.min_depth) {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect::<Vec<_>>()
                };
                let g = Graph::from_tree(root, data, cluster_scorer, self.min_depth);
                mt_log!(
                    Level::Info,
                    "Default Graph created with {} vertices for {} {} under {} {}...",
                    g.cardinality(),
                    data.name(),
                    data.metric().name(),
                    member.name(),
                    model.name()
                );
                g
            })
            .collect()
    }

    /// Parallel version of `generate_training_data`.
    fn par_generate_training_data(&self, y_true: &[f32], graphs: &mut [Graph<I, U, D, S>]) -> TrainingData {
        self.models
            .par_iter()
            .zip(graphs.par_iter_mut())
            .map(|((member, _), graph)| {
                let anomaly_ratings = member.evaluate_points(graph);
                graph
                    .iter_clusters()
                    .map(|v| {
                        let train_x = v.ratios().to_vec();

                        let (y_true, y_pred) = v
                            .indices()
                            .map(|i| (y_true[i], anomaly_ratings[i]))
                            .chain(core::iter::once((0.0, 0.0)))
                            .chain(core::iter::once((1.0, 1.0)))
                            .unzip::<_, _, Vec<_>, Vec<_>>();

                        let roc_auc = roc_auc_score(&y_true, &y_pred).as_f32();

                        (train_x, roc_auc)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Parallel version of `train_models`.
    fn par_train_models(&mut self, training_data: &TrainingData, step_data: &TrainingData) -> Result<(), String> {
        mt_log!(Level::Info, "Training models with {} samples...", training_data.len());

        let training_data = training_data
            .par_iter()
            .zip(step_data.par_iter())
            .map(|(a, b)| a.iter().chain(b.iter()).cloned().collect::<Vec<_>>())
            .collect::<Vec<_>>();

        self.models
            .par_iter_mut()
            .zip(training_data.into_par_iter())
            .try_for_each(|((_, ml_model), ml_data)| {
                let (train_x, train_y) = ml_data.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                ml_model.train(&train_x, &train_y)
            })
    }
}
