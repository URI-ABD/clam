//! Inferring with meta-ml models.

use linfa::prelude::*;
use linfa_linear::{FittedLinearRegression, TweedieRegressor};
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

use crate::chaoda::NUM_RATIOS;

/// A trained meta-ml model.
#[derive(Clone, Serialize, Deserialize)]
pub enum TrainedMetaMlModel {
    /// A linear regression model.
    LinearRegression(FittedLinearRegression<f32>),
    /// A Tweedie regression model.
    TweedieRegression(TweedieRegressor<f32>),
}

impl TrainedMetaMlModel {
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

    /// Predict the suitability of several `Cluster`s for selection in a `Graph`.
    ///
    /// This method is convenient when we want to predict the suitability of several `Cluster`s at once,
    /// and using several `MetaML` models.
    ///
    /// # Arguments
    ///
    /// * `props`: A matrix where each row contains the aggregated anomaly properties of `Cluster`s in a `Graph`.
    ///
    /// # Returns
    ///
    /// The suitability of the `Cluster`s for selection in a `Graph`.
    ///
    /// # Errors
    ///
    /// * If the number of features in the data does not match the number of features in the model.
    /// * If the model cannot predict the data.
    pub fn predict(&self, props: &[f32]) -> Result<Vec<f32>, String> {
        if props.is_empty() || (props.len() % NUM_RATIOS != 0) {
            return Err(format!(
                "Number of features in data ({}) does not match number of features in model ({})",
                props.len(),
                NUM_RATIOS
            ));
        }
        let num_rows = props.len() / NUM_RATIOS;
        let shape = (num_rows, NUM_RATIOS);
        let props = Array2::from_shape_vec(shape, props.to_vec()).map_err(|e| e.to_string())?;

        ftlog::debug!(
            "Predicting with MetaML model {} on {} data samples with {} dims",
            self.name(),
            num_rows,
            NUM_RATIOS,
        );

        let scores = match self {
            Self::LinearRegression(model) => model.predict(&props),
            Self::TweedieRegression(model) => model.predict(&props),
        };

        Ok(scores.to_vec())
    }
}
