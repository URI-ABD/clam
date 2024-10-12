//! Clustered Hierarchical Anomaly and Outlier Detection Algorithms

mod cluster;
mod graph;
mod inference;
mod training;

pub use cluster::{Ratios, Vertex};
use distances::Number;
pub use graph::Graph;
pub use inference::{Chaoda, TrainedMetaMlModel};
#[allow(clippy::module_name_repetitions)]
pub use training::{ChaodaTrainer, GraphAlgorithm, TrainableMetaMlModel};

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

#[cfg(test)]
mod tests {
    use distances::Number;

    #[test]
    fn test_roc_auc_score() {
        let y_score = (0..100).step_by(10).map(|s| s.as_f32() / 100.0).collect::<Vec<_>>();

        let y_true = y_score.iter().map(|&s| s > 0.5).collect::<Vec<_>>();
        let auc = super::roc_auc_score(&y_true, &y_score).unwrap();
        assert_eq!(auc, 1.0);

        let y_true = y_true.into_iter().map(|t| !t).collect::<Vec<_>>();
        let auc = super::roc_auc_score(&y_true, &y_score).unwrap();
        assert_eq!(auc, 0.0);
    }
}
