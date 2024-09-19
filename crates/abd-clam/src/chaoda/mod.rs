//! Clustered Hierarchical Anomaly and Outlier Detection Algorithms

mod cluster;
mod graph;
mod inference;
mod training;

use linfa::{metrics::BinaryClassification, prelude::Pr};

pub use cluster::{Ratios, Vertex};
pub use graph::Graph;
pub use inference::{Chaoda, TrainedMetaMlModel};
pub use training::{ChaodaTrainer, GraphAlgorithm, TrainableMetaMlModel};

const NUM_RATIOS: usize = 6;

/// The area under the receiver operating characteristic curve.
pub fn roc_auc_score(y_true: &[bool], y_score: &[f32]) -> Result<f32, String> {
    let scores = y_score
        .iter()
        .map(|&s| Pr::try_from(s))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let roc_curve =
        <&[Pr] as BinaryClassification<&[bool]>>::roc(&scores.as_slice(), y_true).map_err(|e| e.to_string())?;
    Ok(roc_curve.area_under_curve())
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
