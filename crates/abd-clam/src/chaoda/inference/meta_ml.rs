//! Inferring with meta-ml models.

use linfa_linear::{FittedIsotonicRegression, FittedLinearRegression};
use serde::{Deserialize, Serialize};

/// A trained meta-ml model.
#[derive(Clone, Serialize, Deserialize)]
pub enum TrainedMetaMlModel {
    /// A linear regression model.
    LinearRegression(FittedLinearRegression<f32>),
    /// An Isotonic Regression model.
    IsotonicRegression(FittedIsotonicRegression<f32>),
}

impl TrainedMetaMlModel {
    /// Get the name of the model.
    pub fn name(&self) -> &str {
        match self {
            Self::LinearRegression(_) => "LinearRegression",
            Self::IsotonicRegression(_) => "IsotonicRegression",
        }
    }

    /// Get a short name for the model.
    pub fn short_name(&self) -> &str {
        match self {
            Self::LinearRegression(_) => "LR",
            Self::IsotonicRegression(_) => "IR",
        }
    }

    /// Predict the suitability of several `Cluster`s for selection in a `Graph`.
    ///
    /// This method is convenient when we want to predict the suitability of several `Cluster`s at once,
    /// and using several `MetaML` models.
    ///
    /// # Arguments
    ///
    /// * `data`: A matrix where each row contains the aggregated anomaly properties of `Cluster`s in a `Graph`.
    ///
    /// # Returns
    ///
    /// The suitability of the `Cluster`s for selection in a `Graph`.
    ///
    /// # Errors
    ///
    /// * If the number of features in the data does not match the number of features in the model.
    /// * If the model cannot predict the data.
    pub fn predict(&self, props: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        ftlog::info!(
            "Predicting with MetaML model {} on {} data samples with {} dims",
            self.name(),
            props.len(),
            props[0].len()
        );

        todo!()

        // let data = data.iter().map(Vec::as_slice).collect::<Vec<_>>();
        // let data = DenseMatrix::from_2d_array(&data);
        // match self {
        //     Self::LinearRegression(model) => model.predict(&data),
        //     Self::ElasticNet(model) => model.predict(&data),
        //     Self::Lasso(model) => model.predict(&data),
        //     Self::RidgeRegression(model) => model.predict(&data),
        //     Self::DecisionTreeRegressor(model) => model.predict(&data),
        //     Self::RandomForestRegressor(model) => model.predict(&data),
        // }
        // .map_err(|e| e.to_string())
    }
}
