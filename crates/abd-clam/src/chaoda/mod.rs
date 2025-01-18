//! Clustered Hierarchical Anomaly and Outlier Detection Algorithms

mod cluster;
mod graph;
mod inference;
mod training;

pub use cluster::{Ratios, Vertex};
use distances::Number;
pub use graph::Graph;
pub use inference::{Chaoda, TrainedMetaMlModel, TrainedSmc};
#[allow(clippy::module_name_repetitions)]
pub use training::{ChaodaTrainer, GraphAlgorithm, TrainableMetaMlModel, TrainableSmc};

/// The number of anomaly ratios we use in CHAODA
const NUM_RATIOS: usize = 6;

/// The area under the receiver operating characteristic curve.
///
/// # Arguments
///
/// * `y_true`: The true binary labels.
/// * `y_pred`: The predicted scores.
///
/// # Returns
///
/// The area under the receiver operating characteristic curve.
///
/// # Errors
///
/// * If the number of scores does not match the number of labels.
/// * If the scores cannot be converted to probabilities.
/// * If the ROC curve cannot be calculated.
pub fn roc_auc_score(y_true: &[bool], y_pred: &[f32]) -> Result<f32, String> {
    if y_true.len() != y_pred.len() {
        return Err("The number of scores does not match the number of labels".to_string());
    }
    let y_true = y_true
        .iter()
        .map(|&t| if t { 1_f32 } else { 0_f32 })
        .collect::<Vec<_>>();
    let y_pred = y_pred.to_vec();
    Ok(smartcore::metrics::roc_auc_score(&y_true, &y_pred).as_f32())
}
