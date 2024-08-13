//! Meta Machine Learning models for CHAODA.

use serde::{Deserialize, Serialize};
use smartcore::{
    ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters},
    linalg::basic::matrix::DenseMatrix,
    linear::{
        elastic_net::{ElasticNet, ElasticNetParameters},
        lasso::{Lasso, LassoParameters},
        linear_regression::{LinearRegression, LinearRegressionParameters, LinearRegressionSolverName},
        ridge_regression::{RidgeRegression, RidgeRegressionParameters},
    },
    tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters},
};

/// A Meta Machine Learning model for CHAODA.
#[derive(Serialize, Deserialize)]
pub enum MlModel {
    /// A linear regression model.
    LinearRegression(LinearRegression<f32, f32, DenseMatrix<f32>, Vec<f32>>),
    /// An Elastic Net model.
    ElasticNet(ElasticNet<f32, f32, DenseMatrix<f32>, Vec<f32>>),
    /// A Lasso model.
    Lasso(Lasso<f32, f32, DenseMatrix<f32>, Vec<f32>>),
    /// A Ridge Regression model.
    RidgeRegression(RidgeRegression<f32, f32, DenseMatrix<f32>, Vec<f32>>),
    /// A Decision Tree Regressor model.
    DecisionTreeRegressor(DecisionTreeRegressor<f32, f32, DenseMatrix<f32>, Vec<f32>>),
    /// A Random Forest Regressor model.
    RandomForestRegressor(RandomForestRegressor<f32, f32, DenseMatrix<f32>, Vec<f32>>),
}

impl MlModel {
    /// Create a new `MetaMlModel`.
    ///
    /// # Arguments
    ///
    /// * `model`: The name of the model.
    ///
    /// # Errors
    ///
    /// * If the model name is unknown.
    pub fn new(model: &str) -> Result<Self, String> {
        // Dummy data for model initialization. This comes from the `smartcore` examples.
        let dummy_x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
            &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);
        let dummy_y = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9,
        ];

        Ok(match model {
            "lr" | "LR" | "LinearRegression" => Self::LinearRegression(
                LinearRegression::fit(
                    &dummy_x,
                    &dummy_y,
                    LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::QR),
                )
                .map_err(|e| e.to_string())?,
            ),
            "en" | "EN" | "ElasticNet" => Self::ElasticNet(
                ElasticNet::fit(&dummy_x, &dummy_y, ElasticNetParameters::default()).map_err(|e| e.to_string())?,
            ),
            "la" | "LA" | "Lasso" => {
                Self::Lasso(Lasso::fit(&dummy_x, &dummy_y, LassoParameters::default()).map_err(|e| e.to_string())?)
            }
            "rr" | "RR" | "RidgeRegression" => Self::RidgeRegression(
                RidgeRegression::fit(&dummy_x, &dummy_y, RidgeRegressionParameters::default())
                    .map_err(|e| e.to_string())?,
            ),
            "dt" | "DT" | "DecisionTreeRegressor" => Self::DecisionTreeRegressor(
                DecisionTreeRegressor::fit(&dummy_x, &dummy_y, DecisionTreeRegressorParameters::default())
                    .map_err(|e| e.to_string())?,
            ),
            "rf" | "RF" | "RandomForestRegressor" => Self::RandomForestRegressor(
                RandomForestRegressor::fit(&dummy_x, &dummy_y, RandomForestRegressorParameters::default())
                    .map_err(|e| e.to_string())?,
            ),
            _ => return Err(format!("Unknown model: {model}")),
        })
    }

    /// Get the default models.
    #[must_use]
    pub fn defaults() -> Vec<Self> {
        vec![
            Self::new("LR").unwrap_or_else(|e| unreachable!("{e}")),
            // Self::new("EN").unwrap_or_else(|e| unreachable!("{e}")),
            // Self::new("LA").unwrap_or_else(|e| unreachable!("{e}")),
            // Self::new("RR").unwrap_or_else(|e| unreachable!("{e}")),
            Self::new("DT").unwrap_or_else(|e| unreachable!("{e}")),
            // Self::new("RF").unwrap_or_else(|e| unreachable!("{e}")),
        ]
    }

    /// Train the model on data from a `Graph`.
    ///
    /// # Arguments
    ///
    /// * `data`: A matrix where each row contains the aggregated anomaly properties of `Cluster`s in a `Graph`.
    /// * `roc_scores`: The ROC score for each `Cluster`.
    ///
    /// # Errors
    ///
    /// * If the number of `labels` is not equal to the cardinality of the data.
    pub fn train(&mut self, data: &[Vec<f32>], roc_scores: &Vec<f32>) -> Result<(), String> {
        let data = data.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let data = DenseMatrix::from_2d_array(&data);
        match self {
            Self::LinearRegression(model) => {
                *model = LinearRegression::fit(&data, roc_scores, LinearRegressionParameters::default())
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            Self::ElasticNet(model) => {
                *model =
                    ElasticNet::fit(&data, roc_scores, ElasticNetParameters::default()).map_err(|e| e.to_string())?;
                Ok(())
            }
            Self::Lasso(model) => {
                *model = Lasso::fit(&data, roc_scores, LassoParameters::default()).map_err(|e| e.to_string())?;
                Ok(())
            }
            Self::RidgeRegression(model) => {
                *model = RidgeRegression::fit(&data, roc_scores, RidgeRegressionParameters::default())
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            Self::DecisionTreeRegressor(model) => {
                *model = DecisionTreeRegressor::fit(&data, roc_scores, DecisionTreeRegressorParameters::default())
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            Self::RandomForestRegressor(model) => {
                *model = RandomForestRegressor::fit(&data, roc_scores, RandomForestRegressorParameters::default())
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
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
    pub fn predict(&self, data: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let data = data.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let data = DenseMatrix::from_2d_array(&data);
        match self {
            Self::LinearRegression(model) => model.predict(&data),
            Self::ElasticNet(model) => model.predict(&data),
            Self::Lasso(model) => model.predict(&data),
            Self::RidgeRegression(model) => model.predict(&data),
            Self::DecisionTreeRegressor(model) => model.predict(&data),
            Self::RandomForestRegressor(model) => model.predict(&data),
        }
        .map_err(|e| e.to_string())
    }
}
