//! Functions and traits for training meta-ML prediction algorithms for ranking `Cluster`s before creating `Graph`s.

#![expect(dead_code, unused_imports, unused_variables, unused_mut, unreachable_code)]

use ndarray::prelude::*;

use crate::{DistanceValue, PartitionStrategy, Tree};

use super::{MetaMlModel, prediction::ChaodaPredictor};

mod gen_samples;

pub use gen_samples::{gen_training_sample_single, gen_training_samples_chaoda, gen_training_samples_graphs};

/// A type alias for a metric function that takes two items of type `I` and returns a distance value of type `T`.
type MetricFn<I, T> = Box<dyn Fn(&I, &I) -> T>;

/// A Meta-ML model that can be trained to predict the quality of `Cluster`s for ranking before creating `Graph`s.
pub trait ChaodaTrainer<Params>: ChaodaPredictor + Sized {
    /// Train the model on the given training data.
    ///
    /// # Arguments
    ///
    /// - `features`: Anomaly features for `Graph`s or `Cluster`s, where each row corresponds to a sample and each column corresponds to a feature.
    /// - `roc_scores`: The ROC AUC scores for the corresponding samples, which serve as the target variable for training.
    /// - `parameters`: Hyper-parameters for training the model.
    fn fit(features: &ArrayView2<f64>, roc_scores: &ArrayView1<f64>, parameters: Params) -> Result<Self, String>;
}

/// Trains Meta-ML models.
pub fn train_models<I, T>(items: &[(bool, I)], metrics: Vec<MetricFn<I, T>>) -> Result<Vec<MetaMlModel>, String>
where
    T: DistanceValue,
{
    let items_by_ref = items.iter().map(|(b, item)| (*b, item)).collect::<Vec<_>>();

    let trees = metrics
        .into_iter()
        .map(|metric| Box::new(move |a: &&I, b: &&I| metric(a, b)))
        .map(|metric| Tree::new(items_by_ref.clone(), metric, &PartitionStrategy::default(), &|_| ()).map(Tree::annotate_anomaly_features))
        .collect::<Result<Vec<_>, _>>()?;

    let max_depth = trees.iter().map(Tree::max_depth).max().unwrap_or(0);
    let initial_models = layer_models(max_depth);

    let max_epochs = 10; // TODO(Najib): Tune this based on convergence of the training process.
    let mut epoch = 0;
    let mut trained_models = Vec::new();
    loop {
        todo!();

        epoch += 1;
        if epoch > max_epochs {
            break;
        }
    }

    Ok(trained_models)
}

/// Generates a set of `MetaMlModel::Layer` models for each depth up to the maximum depth of the trees in the training data.
fn layer_models(max_depth: usize) -> Vec<MetaMlModel> {
    // TODO(Najib): Tune the step size based on the distribution of tree depths in the training data.
    let step_size = 5;
    (0..=max_depth).step_by(step_size).map(MetaMlModel::Layer).collect()
}
