//! Functions for generating training samples to train meta-ML algorithms.

use crate::{Dataset, DistanceValue, Tree};

use super::super::{
    super::{Chaoda, Graph},
    Features,
    metrics::roc_auc_score,
};

/// Generates training samples from all `Chaoda` algorithms and all `Graph`s for a given `Tree`.
///
/// # Arguments
///
/// - `tree`: The `Tree` used for creating the `Graph`s.
/// - `graphs`: A list of `Graph`s for which the training samples are being generated.
/// - `algorithms`: A list of `Chaoda` algorithms for which to generate training samples.
///
/// # Returns
///
/// A training sample for each combination of `Graph` and `Chaoda` algorithm. See [`gen_training_samples_chaoda`] for details.
///
/// # Errors
///
/// - If any training sample generation fails. See [`gen_training_samples_chaoda`] for more details on possible errors.
pub fn gen_training_samples_graphs<D, M, T, A>(
    tree: &Tree<D, M, T, (A, Features)>,
    graphs: &[Graph<T>],
    algorithms: &[Chaoda],
) -> Result<Vec<Vec<(Features, f64)>>, String>
where
    D: Dataset<Id = bool>,
    T: DistanceValue,
{
    graphs.iter().map(|graph| gen_training_samples_chaoda(tree, graph, algorithms)).collect()
}

/// Generates training samples from all `Chaoda` algorithms for a given `Tree` and `Graph`.
///
/// # Arguments
///
/// - `tree`: The `Tree` used for creating the `Graph`.
/// - `graph`: The `Graph` for which the training samples are being generated.
/// - `algorithms`: A list of `Chaoda` algorithms for which to generate training samples.
///
/// # Returns
///
/// A training sample for each `Chaoda` algorithm. See [`gen_training_sample_single`] for details on the structure of each training sample.
///
/// # Errors
///
/// - If any training sample generation fails. See [`gen_training_sample_single`] for more details on possible errors.
pub fn gen_training_samples_chaoda<D, T, A, M>(
    tree: &Tree<D, M, T, (A, Features)>,
    graph: &Graph<T>,
    algorithms: &[Chaoda],
) -> Result<Vec<(Features, f64)>, String>
where
    D: Dataset<Id = bool>,
    T: DistanceValue,
{
    algorithms.iter().map(|algorithm| gen_training_sample_single(tree, graph, algorithm)).collect()
}

/// Generates a single training sample from a given `Tree`, `Graph`, and `Chaoda` algorithm.
///
/// The features for the training sample are the mean anomaly features of the `Cluster`s in the `tree` that were selected for creating the `Graph`. The target
/// variable for the training sample is the ROC AUC score of the anomaly scores computed by the `algorithm` for the `tree` and `graph`.
///
/// # Arguments
///
/// - `tree`: The `Tree` used for creating the `Graph`.
/// - `graph`: The `Graph` for which the training sample is being generated.
/// - `algorithm`: The `Chaoda` algorithm used to compute anomaly scores for the `tree` and `graph`.
///
/// # Returns
///
/// A tuple containing:
///
/// - The mean of the anomaly features of the `Cluster`s in the `tree` that were selected for creating the `Graph`.
/// - The ROC AUC score of the anomaly scores computed by the `algorithm` for the `tree` and `graph`.
///
/// # Errors
///
/// - If the `algorithm` fails to compute anomaly scores for the `tree` and `graph`. See [`Chaoda::anomaly_scores`] for more details on possible errors.
/// - If the ROC AUC score cannot be computed from the true labels and predicted scores. See [`roc_auc_score`] for more details on possible errors.
pub fn gen_training_sample_single<D, M, T, A>(tree: &Tree<D, M, T, (A, Features)>, graph: &Graph<T>, algorithm: &Chaoda) -> Result<(Features, f64), String>
where
    D: Dataset<Id = bool>,
    T: DistanceValue,
{
    let y_true = tree.dataset.as_slice().iter().map(|(a, _)| *a);
    algorithm
        .anomaly_scores(graph, tree)
        .and_then(|y_pred| roc_auc_score(y_true, &y_pred))
        .and_then(|score| graph.mean_anomaly_features(tree).map(|features| (features, score)))
}
