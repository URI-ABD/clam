//! Clustered Hierarchical Anomaly and Outlier Detection Algorithms

mod cluster;
mod graph;
mod members;
mod meta_ml;
mod single_metric_model;

use distances::Number;
use mt_logger::{mt_log, Level};
use rayon::prelude::*;

pub use cluster::Vertex;
pub use graph::Graph;
pub use members::{Algorithm, Member};
pub use meta_ml::MlModel;

pub use single_metric_model::TrainingData;
use smartcore::metrics::roc_auc_score;

use crate::{
    adapter::Adapter, cluster::ParCluster, dataset::ParDataset, partition::ParPartition, Cluster, Dataset, Metric,
    Partition,
};
use single_metric_model::SingleMetricModel;

/// The combination of a member and meta-ml models for each metric.
type ModelCombinations<const M: usize> = [Vec<(Member, Vec<MlModel>)>; M];

/// Clustered Hierarchical Anomaly and Outlier Detection Algorithms.
///
/// # Type Parameters
///
/// * `I` - The type of the instances.
/// * `U` - The type of the distances between instances.
/// * `D` - The type of the dataset.
/// * `S` - The type of the source `Cluster` for the `Vertex`es.
/// * `M` - The number of metrics.
pub struct Chaoda<I: Clone, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>, const M: usize> {
    /// The individual models for each metric in use.
    #[allow(dead_code)]
    models: [SingleMetricModel<I, U, D, S>; M],
}

impl<I: Clone, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>, const M: usize> Chaoda<I, U, D, S, M> {
    /// Create new trees for use with CHAODA.
    ///
    /// # Arguments
    ///
    /// * `datasets` - The datasets to use for building the trees.
    /// * `criteria` - The criteria for partitioning clusters, one for each dataset.
    /// * `metrics` - The metrics to use for each dataset.
    ///
    /// # Type Parameters
    ///
    /// * `N` - The number of datasets.
    /// * `M` - The number of metrics.
    /// * `C` - The criteria for partitioning clusters.
    ///
    /// # Returns
    ///
    /// For each combination of dataset and metric, the root of the tree.
    #[must_use]
    pub fn new_trees<const N: usize, C: Fn(&S) -> bool>(
        datasets: &mut [D; N],
        criteria: &[C; N],
        metrics: &[Metric<I, U>; M],
        seed: Option<u64>,
    ) -> [[Vertex<I, U, D, S>; M]; N]
    where
        S: Partition<I, U, D>,
    {
        let mut trees = Vec::new();

        for (data, criteria) in datasets.iter_mut().zip(criteria) {
            let mut roots = Vec::new();
            for metric in metrics {
                data.set_metric(metric.clone());
                mt_log!(
                    Level::Info,
                    "Building tree on {} for metric: {}",
                    data.name(),
                    metric.name()
                );

                let source = S::new_tree(data, criteria, seed);
                let root = Vertex::adapt_tree(source, None);
                roots.push(root);
            }

            let roots: [Vertex<I, U, D, S>; M] = roots
                .try_into()
                .unwrap_or_else(|_| unreachable!("We built a tree for each metric."));
            trees.push(roots);
        }

        trees
            .try_into()
            .unwrap_or_else(|_| unreachable!("We built trees for each dataset."))
    }

    /// Returns the default model combinations for CHAODA.
    #[must_use]
    pub fn default_model_combinations() -> ModelCombinations<M> {
        let mut combinations = Vec::new();

        for _ in 0..M {
            let n_combinations = Member::default_members()
                .into_iter()
                .map(|member| (member, MlModel::default_models()))
                .collect();
            combinations.push(n_combinations);
        }

        combinations
            .try_into()
            .unwrap_or_else(|_| unreachable!("We built model combinations for each N."))
    }

    /// Create a new CHAODA instance.
    ///
    /// # Arguments
    ///
    /// * `metrics` - The metrics to use in the ensemble.
    /// * `model_combinations` - The model combinations to use for each metric.
    /// * `min_depth` - The minimum depth of the `Vertex`es to consider.
    #[must_use]
    pub fn new(metrics: &[Metric<I, U>; M], model_combinations: Option<ModelCombinations<M>>, min_depth: usize) -> Self {
        let model_combinations = model_combinations.unwrap_or_else(|| Self::default_model_combinations());

        let models = metrics
            .iter()
            .zip(model_combinations)
            .map(|(metric, model_combinations)| SingleMetricModel::new(metric.clone(), model_combinations, min_depth))
            .collect::<Vec<_>>();

        let models = models
            .try_into()
            .unwrap_or_else(|_| unreachable!("We built a model for each metric."));

        Self { models }
    }

    /// Train the model.
    ///
    /// # Arguments
    ///
    /// * `datasets` - The datasets to use for training.
    /// * `trees` - The trees to use for training, one for each combination of
    ///   dataset and metric.
    /// * `labels` - The labels for the instances in the datasets.
    /// * `num_epochs` - The number of epochs to train for.
    /// * `previous_data` - The previous training data to use.
    ///
    /// # Type Parameters
    ///
    /// * `N` - The number of datasets.
    ///
    /// # Returns
    ///
    /// The training data after training.
    ///
    /// # Errors
    ///
    /// - If the number of labels does not match the number of instances in the
    ///   dataset.
    /// - If the training of the meta-ml models fails.
    pub fn train<const N: usize>(
        &mut self,
        datasets: &mut [D; N],
        trees: &[[Vertex<I, U, D, S>; M]; N],
        labels: &[Vec<bool>; N],
        num_epochs: usize,
        previous_data: Option<TrainingData>,
    ) -> Result<TrainingData, String> {
        let mut training_data = previous_data.unwrap_or_default();
        mt_log!(
            Level::Info,
            "Training CHAODA on {M} metrics starting with {} training samples...",
            training_data.len()
        );

        for e in 0..num_epochs {
            mt_log!(
                Level::Info,
                "Training epoch {}/{num_epochs}, with {} samples...",
                e + 1,
                training_data.iter().map(Vec::len).sum::<usize>()
            );

            let epoch_data = datasets
                .iter_mut()
                .zip(labels.iter())
                .zip(trees.iter())
                .flat_map(|((data, labels), roots)| {
                    let ep_data = self
                        .models
                        .iter_mut()
                        .zip(roots.iter())
                        .flat_map(|(model, root)| {
                            model
                                .train_step(data, root, labels, &training_data)
                                .unwrap_or_else(|e| unreachable!("{e}"))
                        })
                        .collect::<Vec<_>>();

                    let roc_score = self.evaluate(data, roots, labels);
                    mt_log!(
                        Level::Info,
                        "Epoch {}/{num_epochs}, data: {}, roc-score: {roc_score:.6}",
                        e + 1,
                        data.name()
                    );
                    ep_data
                })
                .collect::<Vec<_>>();

            training_data.extend(epoch_data);
        }

        Ok(training_data)
    }

    /// Predict the anomaly scores for the instances in the dataset.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to predict on.
    /// * `trees` - The trees to use for prediction, one for each metric.
    ///
    /// # Returns
    ///
    /// The anomaly scores for the instances in the dataset, aggregated across
    /// the metrics.
    ///
    /// # Panics
    ///
    /// - TODO: Remove this panic.
    pub fn predict(&self, data: &mut D, trees: &[Vertex<I, U, D, S>; M]) -> Vec<f32> {
        let scores = self
            .models
            .iter()
            .zip(trees.iter())
            .flat_map(|(model, root)| model.predict(data, root))
            .collect::<Vec<_>>();

        // Convert the scores to column-major order.
        let mut col_scores = vec![vec![0.0; M]; data.cardinality()];
        for (r, row) in scores.into_iter().enumerate() {
            for (c, s) in row.into_iter().enumerate() {
                col_scores[r][c] = s;
            }
        }

        // Calculate the mean of the scores for each column.
        let scores = col_scores.iter().map(|col| crate::utils::mean(col)).collect::<Vec<_>>();

        assert_eq!(scores.len(), data.cardinality());
        scores
    }

    /// Evaluate the model using roc-auc score on the dataset.
    pub fn evaluate(&self, data: &mut D, trees: &[Vertex<I, U, D, S>; M], labels: &[bool]) -> f32 {
        let scores = self.predict(data, trees);
        let y_true = labels.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect::<Vec<_>>();
        roc_auc_score(&y_true, &scores).as_f32()
    }

    /// Save the model to a file.
    ///
    /// # Parameters
    ///
    /// * `path` - The path to the file to save the model to.
    ///
    /// # Errors
    ///
    /// - If the model cannot be serialized.
    /// - If the file cannot be written to.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), String> {
        let mut bytes = Vec::new();
        for model in &self.models {
            let model_bytes = bincode::serialize(model).map_err(|e| e.to_string())?;
            bytes.extend(model_bytes.len().to_le_bytes());
            bytes.extend(model_bytes);
        }
        // write the bytes to the file
        std::fs::write(path, bytes).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Load the model from a file.
    ///
    /// # Parameters
    ///
    /// * `path` - The path to the file to load the model from.
    /// * `metrics` - The metrics to use for the model.
    ///
    /// # Errors
    ///
    /// - If the number of models does not match the number of metrics.
    /// - If the model cannot be deserialized.
    pub fn load<P: AsRef<std::path::Path>>(path: P, metrics: &[Metric<I, U>; M]) -> Result<Self, String>
    where
        I: Default,
    {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let mut models = Vec::new();
        let mut offset = 0;
        while offset < bytes.len() {
            let model = crate::cakes::codec::read_encoding(&bytes, &mut offset);
            let model = bincode::deserialize::<SingleMetricModel<I, U, D, S>>(&model).map_err(|e| e.to_string())?;
            models.push(model);
        }

        if models.len() != M {
            return Err(format!(
                "The number of models ({}) does not match the number of metrics ({})",
                models.len(),
                M
            ));
        }

        let models = models
            .into_iter()
            .zip(metrics.iter())
            .map(|(model, metric)| model.with_metric(metric.clone()))
            .collect::<Vec<_>>();

        let models = models
            .try_into()
            .unwrap_or_else(|_| unreachable!("Failed to convert models into an array of {M} elements."));

        Ok(Self { models })
    }
}

impl<I: Clone + Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<I, U, D>, const M: usize>
    Chaoda<I, U, D, S, M>
{
    /// Parallel version of `new_trees`.
    #[must_use]
    pub fn par_new_trees<const N: usize, C: (Fn(&S) -> bool) + Send + Sync>(
        datasets: &mut [D; N],
        criteria: &[C; N],
        metrics: &[Metric<I, U>; M],
        seed: Option<u64>,
    ) -> [[Vertex<I, U, D, S>; M]; N]
    where
        S: ParPartition<I, U, D>,
    {
        let mut trees = Vec::new();

        for (data, criteria) in datasets.iter_mut().zip(criteria) {
            let mut roots = Vec::new();
            for metric in metrics {
                data.set_metric(metric.clone());

                mt_log!(
                    Level::Info,
                    "Building tree on {} for metric: {}",
                    data.name(),
                    metric.name()
                );

                let source = S::par_new_tree(data, criteria, seed);
                let root = Vertex::adapt_tree(source, None);
                roots.push(root);
            }

            let roots: [Vertex<I, U, D, S>; M] = roots
                .try_into()
                .unwrap_or_else(|_| unreachable!("We built a tree for each metric."));
            trees.push(roots);
        }

        trees
            .try_into()
            .unwrap_or_else(|_| unreachable!("We built trees for each dataset."))
    }

    /// Parallel version of `train`.
    #[allow(clippy::missing_errors_doc)]
    pub fn par_train<const N: usize>(
        &mut self,
        datasets: &mut [D; N],
        trees: &[[Vertex<I, U, D, S>; M]; N],
        labels: &[Vec<bool>; N],
        num_epochs: usize,
        previous_data: Option<TrainingData>,
    ) -> Result<TrainingData, String> {
        let mut training_data = previous_data.unwrap_or_default();
        mt_log!(
            Level::Info,
            "Training CHAODA on {M} metrics starting with {} training samples...",
            training_data.len()
        );

        for e in 0..num_epochs {
            mt_log!(
                Level::Info,
                "Training epoch {}/{num_epochs}, with {} samples...",
                e + 1,
                training_data.iter().map(Vec::len).sum::<usize>()
            );

            let epoch_data = datasets
                .iter_mut()
                .zip(labels.iter())
                .zip(trees.iter())
                .flat_map(|((data, labels), roots)| {
                    let ep_data = self
                        .models
                        .iter_mut()
                        .zip(roots.iter())
                        .flat_map(|(model, root)| {
                            model
                                .par_train_step(data, root, labels, &training_data)
                                .unwrap_or_else(|e| unreachable!("{e}"))
                        })
                        .collect::<Vec<_>>();

                    let roc_score = self.par_evaluate(data, roots, labels);
                    mt_log!(
                        Level::Info,
                        "Epoch {}/{num_epochs}, data: {}, roc-score: {roc_score:.6}",
                        e + 1,
                        data.name()
                    );
                    ep_data
                })
                .collect::<Vec<_>>();

            training_data.extend(epoch_data);
        }

        Ok(training_data)
    }

    /// Parallel version of `train`.
    pub fn par_predict(&self, data: &mut D, trees: &[Vertex<I, U, D, S>; M]) -> Vec<f32> {
        let scores = self
            .models
            .iter()
            .zip(trees.iter())
            .flat_map(|(model, root)| model.par_predict(data, root))
            .collect::<Vec<_>>();

        // Convert the scores to column-major order.
        let mut col_scores = vec![vec![0.0; M]; data.cardinality()];
        for (r, row) in scores.into_iter().enumerate() {
            for (c, s) in row.into_iter().enumerate() {
                col_scores[r][c] = s;
            }
        }

        // Calculate the mean of the scores for each column.
        col_scores.par_iter().map(|col| crate::utils::mean(col)).collect()
    }

    /// Parallel version of `train`.
    pub fn par_evaluate(&self, data: &mut D, trees: &[Vertex<I, U, D, S>; M], labels: &[bool]) -> f32 {
        let scores = self.par_predict(data, trees);
        let y_true = labels.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect::<Vec<_>>();
        roc_auc_score(&y_true, &scores).as_f32()
    }
}
