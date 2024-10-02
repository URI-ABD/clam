//! Meta Machine Learning models for CHAODA.

use smartcore::{
    linalg::basic::matrix::DenseMatrix,
    linear::linear_regression::{LinearRegression, LinearRegressionParameters},
    tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters},
};

/// A meta machine learning model.
#[derive(Clone)]
pub enum TrainableMetaMlModel {
    /// A linear regression model.
    LinearRegression(LinearRegressionParameters),
    /// A Decision Tree model.
    DecisionTree(DecisionTreeRegressorParameters),
}

impl Default for TrainableMetaMlModel {
    fn default() -> Self {
        Self::LinearRegression(LinearRegressionParameters::default())
    }
}

impl TrainableMetaMlModel {
    /// Get the default models.
    #[must_use]
    pub fn default_models() -> Vec<Self> {
        vec![
            Self::LinearRegression(LinearRegressionParameters::default()),
            Self::DecisionTree(DecisionTreeRegressorParameters::default()),
        ]
    }

    /// Get the name of the model.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::LinearRegression(_) => "LinearRegression",
            Self::DecisionTree(_) => "DecisionTree",
        }
    }

    /// Get a short name for the model.
    #[must_use]
    pub const fn short_name(&self) -> &str {
        match self {
            Self::LinearRegression(_) => "LR",
            Self::DecisionTree(_) => "DT",
        }
    }

    /// Train the model on the given data.
    ///
    /// # Arguments
    ///
    /// - `num_features`: The number of features in the data.
    /// - `samples`: The samples to train on, in row-major order.
    /// - `scores`: The scores to train on.
    ///
    /// # Returns
    ///
    /// The trained model.
    ///
    /// # Errors
    ///
    /// - If the number of samples is zero.
    /// - If the number of samples is not a multiple of the number of features.
    /// - If the number of samples-rows and scores do not match.
    /// - If the model cannot be trained.
    pub fn train(
        &self,
        num_features: usize,
        samples: &[f32],
        scores: &[f32],
    ) -> Result<crate::chaoda::TrainedMetaMlModel, String> {
        if samples.is_empty() {
            return Err("No samples provided".to_string());
        }
        if samples.len() % num_features != 0 {
            return Err("Length of samples is not a multiple of the number of features".to_string());
        }

        if (samples.len() / num_features) != scores.len() {
            return Err("Number of samples and scores do not match".to_string());
        }

        let samples = samples
            .chunks_exact(num_features)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();
        let samples =
            DenseMatrix::from_2d_vec(&samples).map_err(|e| format!("Failed to create matrix of samples: {e}"))?;
        let scores = scores.to_vec();

        match self {
            Self::LinearRegression(params) => {
                let model = LinearRegression::fit(&samples, &scores, params.clone())
                    .map_err(|e| format!("Failed to train model: {e}"))?;
                Ok(crate::chaoda::TrainedMetaMlModel::LinearRegression(model))
            }
            Self::DecisionTree(params) => {
                let model = DecisionTreeRegressor::fit(&samples, &scores, params.clone())
                    .map_err(|e| format!("Failed to train model: {e}"))?;
                Ok(crate::chaoda::TrainedMetaMlModel::DecisionTree(model))
            }
        }
    }
}

impl TryFrom<&str> for TrainableMetaMlModel {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "LR" | "lr" | "LinearRegression" => Ok(Self::LinearRegression(LinearRegressionParameters::default())),
            "DT" | "dt" | "DecisionTree" => Ok(Self::DecisionTree(DecisionTreeRegressorParameters::default())),
            _ => Err(format!("Unknown model: {value}")),
        }
    }
}
