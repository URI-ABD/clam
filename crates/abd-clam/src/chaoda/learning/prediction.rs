//! Prediction algorithms for ranking `Cluster`s before creating `Graph`s.

use crate::Cluster;

use super::Features;

/// Meta-ML prediction algorithm for ranking `Cluster`s in a `Tree` before creating `Graph`s.
pub trait ChaodaPredictor {
    /// Uses anomaly features to predict the ROC AUC score as a ranking for `Cluster`s in a `Tree`.
    ///
    /// # Errors
    ///
    /// If the prediction fails.
    #[expect(dead_code)]
    fn predict(&self, features: &Features) -> Result<f64, String>;
}

/// Various Meta-ML models for predicting the quality of `Cluster`s for ranking before creating `Graph`s.
pub enum MetaMlModel {
    /// Assigns a score of 1.0 to all clusters at the given depth and to leaves at a shallower depth, and a score of 0.0 to all other clusters.
    Layer(usize),
    /// Uses a simple linear regression model to predict the ROC AUC score based on the anomaly features.
    LinearRegression,
    /// Uses a decision tree regression model to predict the ROC AUC score based on the anomaly features.
    DecisionTree,
}

impl MetaMlModel {
    /// Predicts the ROC AUC score for a `Cluster` based on its anomaly features using the specified Meta-ML model.
    ///
    /// # Arguments
    ///
    /// - `cluster`: The `Cluster` for which to predict the ROC AUC score.
    ///
    /// # Errors
    ///
    /// If the prediction fails.
    pub fn predict<T, A>(&self, cluster: &Cluster<T, (A, Features)>) -> Result<f64, String> {
        match self {
            &Self::Layer(depth) => {
                if cluster.depth == depth || (cluster.is_leaf() && cluster.depth < depth) {
                    Ok(1.0)
                } else {
                    Ok(0.0)
                }
            }
            _ => todo!(),
        }
    }
}
