//! Tests for the `chaoda` module.

use abd_clam::chaoda::roc_auc_score;
use distances::Number;

#[test]
fn test_roc_auc_score() -> Result<(), String> {
    let y_score = (0..100).step_by(10).map(|s| s.as_f32() / 100.0).collect::<Vec<_>>();

    let y_true = y_score.iter().map(|&s| s > 0.5).collect::<Vec<_>>();
    let auc = roc_auc_score(&y_true, &y_score)?;
    float_cmp::approx_eq!(f32, auc, 1.0);

    let y_true = y_true.into_iter().map(|t| !t).collect::<Vec<_>>();
    let auc = roc_auc_score(&y_true, &y_score)?;
    float_cmp::approx_eq!(f32, auc, 0.0);

    Ok(())
}
