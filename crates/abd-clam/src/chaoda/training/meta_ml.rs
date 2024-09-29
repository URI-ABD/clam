//! Meta Machine Learning models for CHAODA.

use linfa::prelude::*;
use linfa_linear::{LinearRegression, TweedieRegressor, TweedieRegressorParams};
use ndarray::prelude::*;

/// A meta machine learning model.
#[derive(Clone)]
pub enum TrainableMetaMlModel {
    /// A linear regression model.
    LinearRegression(LinearRegression),
    /// An Isotonic Regression model.
    TweedieRegression(TweedieRegressorParams<f32>),
}

impl Default for TrainableMetaMlModel {
    fn default() -> Self {
        Self::LinearRegression(LinearRegression::default())
    }
}

impl TrainableMetaMlModel {
    /// Get the default models.
    #[must_use]
    pub fn default_models() -> Vec<Self> {
        vec![
            Self::LinearRegression(LinearRegression::default()),
            Self::TweedieRegression(TweedieRegressor::params().power(0.0)),
        ]
    }

    /// Get the name of the model.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::LinearRegression(_) => "LinearRegression",
            Self::TweedieRegression(_) => "TweedieRegression",
        }
    }

    /// Get a short name for the model.
    #[must_use]
    pub const fn short_name(&self) -> &str {
        match self {
            Self::LinearRegression(_) => "LR",
            Self::TweedieRegression(_) => "TR",
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
        let num_samples = samples.len() / num_features;
        if num_samples == 0 {
            return Err("Number of samples is zero".to_string());
        }
        if num_samples != scores.len() {
            return Err("Number of samples and scores do not match".to_string());
        }

        let shape = (num_samples, num_features);
        let samples = Array2::from_shape_vec(shape, samples.to_vec())
            .map_err(|e| format!("Failed to create array of samples: {e}"))?;
        let scores = Array1::from(scores.to_vec());
        let data = linfa::Dataset::new(samples, scores);

        match self {
            Self::LinearRegression(model) => {
                let model = model.fit(&data).map_err(|e| format!("Failed to train model: {e}"))?;
                Ok(crate::chaoda::TrainedMetaMlModel::LinearRegression(model))
            }
            Self::TweedieRegression(params) => {
                let model = params.fit(&data).map_err(|e| format!("Failed to train model: {e}"))?;
                Ok(crate::chaoda::TrainedMetaMlModel::TweedieRegression(model))
            }
        }
    }
}

impl TryFrom<&str> for TrainableMetaMlModel {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "LR" | "lr" | "LinearRegression" => Ok(Self::LinearRegression(LinearRegression::default())),
            "TR" | "tr" | "TweedieRegression" => Ok(Self::TweedieRegression(TweedieRegressor::params())),
            _ => Err(format!("Unknown model: {value}")),
        }
    }
}
