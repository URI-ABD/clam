//! Performance metrics for anomaly detection algorithms.

/// Compute the ROC AUC score for binary classification.
///
/// # Arguments
///
/// * `y_true` - An iterator of boolean values representing the true binary labels.
/// * `y_pred` - A vector of floating-point values representing the predicted probabilities for the positive class.
///
/// # Returns
///
/// The ROC AUC score as a floating-point value between 0.0 and 1.0.
///
/// # Errors
///
/// - If the lengths of `y_true` and `y_pred` do not match.
pub fn roc_auc_score<Ids>(y_true: Ids, y_pred: &Vec<f64>) -> Result<f64, String>
where
    Ids: Iterator<Item = bool>,
{
    let y_true = y_true.map(|b| if b { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    if y_true.len() == y_pred.len() {
        Ok(smartcore::metrics::roc_auc_score(&y_true, y_pred))
    } else {
        Err(format!(
            "Length mismatch: y_true has length {}, but y_pred has length {}",
            y_true.len(),
            y_pred.len()
        ))
    }
}
