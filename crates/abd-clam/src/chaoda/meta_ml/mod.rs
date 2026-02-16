//! Meta-Machine Learning (Meta-ML) models for use in `CHAODA`.

use crate::Cluster;

use super::AnomalyFeatures;

mod decision_tree;
mod linear_regression;

pub use decision_tree::DecisionTree;
pub use linear_regression::LinearRegression;

/// A Meta-ML model for use in `CHAODA`.
#[derive(Debug)]
#[non_exhaustive]
#[must_use]
pub enum MetaMlModel {
    /// A decision tree model.
    DecisionTree(DecisionTree),
    /// A linear regression model.
    LinearRegression(LinearRegression),
    /// A model that selects clusters of a given depth in the tree.
    Layer(Layer),
}

impl MetaMlModel {
    /// Predict the quality of a `Cluster` for use in a `Graph` based on its anomaly features.
    ///
    /// # Errors
    ///
    /// - If the model fails to make a prediction.
    pub fn predict<T, A>(&self, cluster: &Cluster<T, (A, AnomalyFeatures)>) -> Result<f64, String> {
        match self {
            Self::DecisionTree(model) => model.predict(cluster),
            Self::LinearRegression(model) => model.predict(cluster),
            Self::Layer(model) => model.predict(cluster),
        }
    }

    /// Train a linear regression model on the given training data.
    ///
    /// # Errors
    ///
    /// - If the training data is invalid or the model cannot be trained.
    pub fn fit_linear_regression(data: &[(AnomalyFeatures, f64)]) -> Result<Self, String> {
        LinearRegression::fit(data).map(Self::LinearRegression)
    }

    /// Train a decision tree model on the given training data.
    ///
    /// # Errors
    ///
    /// - If the training data is invalid or the model cannot be trained.
    pub fn fit_decision_tree(data: &[(AnomalyFeatures, f64)]) -> Result<Self, String> {
        DecisionTree::fit(data).map(Self::DecisionTree)
    }

    /// Create a new `Layer` model that selects clusters of the given depth.
    pub const fn new_layer(depth: usize) -> Self {
        Self::Layer(Layer::new(depth))
    }
}

/// A trait for models that can predict the quality of a `Cluster` for use in a `Graph` based on its anomaly features.
///
/// Higher scores indicate that the `Cluster` should be used in the `Graph`, while lower scores indicate that it should not be used.
trait MetaMlPredictor: Send + Sync {
    /// Predict the quality of a `Cluster` for use in a `Graph` based on its anomaly features.
    ///
    /// # Errors
    ///
    /// - If the model fails to make a prediction.
    fn predict<T, A>(&self, cluster: &Cluster<T, (A, AnomalyFeatures)>) -> Result<f64, String>;
}

/// A trait for models that can be trained to predict the quality of a `Cluster` for use in a `Graph` based on its anomaly features.
trait MetaMlTrainer: MetaMlPredictor + Sized {
    /// Train the model on the given training data.
    ///
    /// # Errors
    ///
    /// - If the training data is invalid or the model cannot be trained.
    fn fit(data: &[(AnomalyFeatures, f64)]) -> Result<Self, String>;
}

/// A simple meta-ml model that selects clusters of a given depth in the tree.
#[must_use]
#[derive(Debug, Clone)]
pub struct Layer {
    /// The depth of the clusters to select.
    depth: usize,
}

impl From<Layer> for super::MetaMlModel {
    fn from(alg: Layer) -> Self {
        Self::Layer(alg)
    }
}

impl Layer {
    /// Create a new `Layer` model that selects clusters of the given depth.
    pub const fn new(depth: usize) -> Self {
        Self { depth }
    }
}

impl MetaMlPredictor for Layer {
    fn predict<T, A>(&self, cluster: &Cluster<T, (A, AnomalyFeatures)>) -> Result<f64, String> {
        let y = if cluster.depth == self.depth { 1.0 } else { 0.0 };
        Ok(y)
    }
}
