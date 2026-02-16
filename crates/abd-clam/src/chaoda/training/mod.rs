//! Functions and traits for training meta-ML prediction algorithms for ranking `Cluster`s before creating `Graph`s.

use rayon::prelude::*;

use crate::{DistanceValue, Tree, chaoda::AnomalyFeatures};

use super::{Graph, algorithms, meta_ml::MetaMlModel};

mod gen_samples;

pub use gen_samples::{gen_training_sample, par_gen_training_sample};

/// A trait for types that can be used as anomaly labels in the training process.
///
/// We provide an implementation of this trait for `bool`.
pub trait AnomalyLabel {
    /// Returns whether the label indicates that the item is an anomaly.
    fn is_anomaly(&self) -> bool;
}

impl AnomalyLabel for bool {
    fn is_anomaly(&self) -> bool {
        *self
    }
}

/// A type-alias for the nested vector of trained Meta-ML models.
type TrainedModels = Vec<(algorithms::GraphAlgorithm, Vec<MetaMlModel>)>;

/// Data to use for training meta-ml models.
#[derive(Debug)]
pub struct MetaMlTrainingData {
    /// The graph algorithms for which we are training meta-ml models.
    algs: Vec<algorithms::GraphAlgorithm>,
    /// A nested vector of training data, where each pair consists of the mean anomaly features for a graph and the associated quality score.
    data: Vec<Vec<(AnomalyFeatures, f64)>>,
}

impl MetaMlTrainingData {
    /// Creates a new `MetaMlTrainingData` instance with the given graph algorithms.
    pub fn new(algs: Vec<algorithms::GraphAlgorithm>) -> Self {
        let data = vec![Vec::new(); algs.len()];
        Self { algs, data }
    }

    /// Train new meta-ml models using the accumulated training data.
    pub fn train_models(&self) -> Result<TrainedModels, String> {
        let trained_models = self
            .data
            .iter()
            .map(|alg_data| Ok(vec![MetaMlModel::fit_linear_regression(alg_data)?, MetaMlModel::fit_decision_tree(alg_data)?]))
            .collect::<Result<Vec<_>, String>>()?;

        Ok(self.algs.iter().cloned().zip(trained_models).collect())
    }

    /// Parallel version of [`train_models`].
    pub fn par_train_models(&self) -> Result<TrainedModels, String> {
        let trained_models = self
            .data
            .par_iter()
            .map(|alg_data| {
                let (lr_model, dt_model) = rayon::join(|| MetaMlModel::fit_linear_regression(alg_data), || MetaMlModel::fit_decision_tree(alg_data));
                Ok(vec![lr_model?, dt_model?])
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(self.algs.iter().cloned().zip(trained_models).collect())
    }

    /// Extracts training data from the given tree.
    ///
    /// # Arguments
    ///
    /// - `tree`: A pre-built tree to use for training the models. The tree will be annotated with anomaly features before training and de-annotated just before
    ///   returning from this function.
    ///
    /// # Returns
    ///
    /// The restored tree and the trained meta-ml models as a nested vector: [`TrainedModels<T, A>`].
    ///
    /// # Errors
    ///
    /// - If the ROC scores fail to compute.
    /// - Training any model fails.
    pub fn extract_data<Id, I, T, A, M>(&mut self, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<(), String>
    where
        Id: AnomalyLabel,
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
    {
        // TODO(Najib): Consider exposing the min_depth and step_size as parameters to this function.
        let min_depth = 4;
        let step_size = 4;
        let tree_depth = tree.max_depth();

        // Use layer graphs to bootstrap training.
        let layer_graphs = (min_depth..=tree_depth)
            .step_by(step_size)
            .map(MetaMlModel::new_layer)
            .map(|predictor| {
                let (directly_selected, ancestors) = tree.select_chaoda_clusters(&predictor, min_depth);
                Graph::from_tree(tree, &directly_selected, &ancestors)
            })
            .collect::<Vec<_>>();

        // Generate the first batch of training data using the layer graphs.
        let mut training_data = self
            .algs
            .iter()
            .map(|alg| {
                layer_graphs
                    .iter()
                    .map(|graph| gen_training_sample(tree, graph, alg))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Over the course of several epochs, train new models and accumulate more training data for each graph algorithm and meta-ml model.
        let max_epochs = 10; // TODO(Najib): Set the number of epochs based on convergence of the training process, after testing with a few datasets.
        for epoch in 0..max_epochs {
            let mut roc_scores = vec![];

            for (alg, alg_data) in self.algs.iter().zip(&mut training_data) {
                // For each algorithm, fit new models using the data we have so far.
                let models = [MetaMlModel::fit_linear_regression(alg_data)?, MetaMlModel::fit_decision_tree(alg_data)?];

                // Generate more data for use in the next epoch
                for model in &models {
                    // For each model, create a new graph, apply the algorithm, and create training data for the next epoch.
                    let (directly_selected, ancestors) = tree.select_chaoda_clusters(model, min_depth);
                    let graph = Graph::from_tree(tree, &directly_selected, &ancestors);
                    let new_sample = gen_training_sample(tree, &graph, alg)?;

                    roc_scores.push(new_sample.1);

                    alg_data.push(new_sample);
                }
            }

            #[expect(clippy::cast_precision_loss)]
            let mean_score = roc_scores.iter().sum::<f64>() / (roc_scores.len() as f64);
            #[expect(clippy::cast_precision_loss)]
            let std_score = roc_scores.iter().map(|&s| s - mean_score).map(|s| s * s).sum::<f64>().sqrt() / (roc_scores.len() as f64);

            ftlog::info!("Epoch {}/{max_epochs}: roc_scores: {mean_score:.2e} +/- {std_score:.2e}", epoch + 1);
        }

        self.append_data(training_data);

        Ok(())
    }

    /// Parallel version of [`extract_data`].
    ///
    /// # Errors
    ///
    /// - See [`extract_data`] for possible errors.
    pub fn par_extract_data<Id, I, T, A, M>(&mut self, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<(), String>
    where
        Id: AnomalyLabel + Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
    {
        let min_depth = 4;
        let step_size = 4;
        let tree_depth = tree.max_depth();

        // Use layer graphs to bootstrap training.
        let layer_depths = (min_depth..=tree_depth).step_by(step_size).collect::<Vec<_>>();
        let layer_graphs = layer_depths
            .into_par_iter()
            .map(MetaMlModel::new_layer)
            .map(|predictor| {
                let (directly_selected, ancestors) = tree.par_select_chaoda_clusters(&predictor, min_depth);
                Graph::par_from_tree(tree, &directly_selected, &ancestors)
            })
            .collect::<Vec<_>>();

        // Generate the first batch of training data using the layer graphs.
        let mut training_data = self
            .algs
            .par_iter()
            .map(|alg| {
                layer_graphs
                    .par_iter()
                    .map(|graph| par_gen_training_sample(tree, graph, alg))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let max_epochs = 10; // TODO(Najib): Tune this based on convergence of the training process.
        for epoch in 0..max_epochs {
            let mut roc_scores = vec![];

            let new_scores = self
                .algs
                .par_iter()
                .zip(&mut training_data)
                .map(|(alg, alg_data)| {
                    // For each algorithm, train new models using the data we have so far.
                    let (lr_model, dt_model) = rayon::join(|| MetaMlModel::fit_linear_regression(alg_data), || MetaMlModel::fit_decision_tree(alg_data));

                    [lr_model?, dt_model?]
                        .iter()
                        .map(|model| {
                            // For each model, create a new graph, apply the algorithm, and create training data for the next epoch.
                            let (directly_selected, ancestors) = tree.par_select_chaoda_clusters(model, min_depth);
                            let graph = Graph::par_from_tree(tree, &directly_selected, &ancestors);
                            let new_sample = par_gen_training_sample(tree, &graph, alg)?;
                            alg_data.push(new_sample);
                            Ok(new_sample.1)
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, String>>()?;

            roc_scores.extend(new_scores.into_iter().flatten());

            #[expect(clippy::cast_precision_loss)]
            let mean_score = roc_scores.iter().sum::<f64>() / (roc_scores.len() as f64);
            #[expect(clippy::cast_precision_loss)]
            let std_score = roc_scores.iter().map(|&s| s - mean_score).map(|s| s * s).sum::<f64>().sqrt() / (roc_scores.len() as f64);

            ftlog::info!("Epoch {}/{max_epochs}: roc_scores: {mean_score:.2e} +/- {std_score:.2e}", epoch + 1);
        }

        self.append_data(training_data);

        Ok(())
    }

    /// Appends the given training data to the existing training data.
    fn append_data(&mut self, new_data: Vec<Vec<(AnomalyFeatures, f64)>>) {
        for (acc, new) in self.data.iter_mut().zip(new_data) {
            acc.extend(new);
        }
    }
}
