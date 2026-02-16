//! Inferring with meta-ml models.

use smartcore::{
    linalg::basic::matrix::DenseMatrix, linear::linear_regression::LinearRegression,
    tree::decision_tree_regressor::DecisionTreeRegressor,
};

use crate::{chaoda::Vertex, DistanceValue};

/// A trained meta-ml model.
#[derive(serde::Serialize, serde::Deserialize)]
pub enum TrainedMetaMlModel {
    /// A linear regression model.
    LinearRegression(LinearRegression<f32, f32, DenseMatrix<f32>, Vec<f32>>),
    /// A Decision Tree model.
    DecisionTree(DecisionTreeRegressor<f32, f32, DenseMatrix<f32>, Vec<f32>>),
}

impl TrainedMetaMlModel {
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
    pub fn predict<T: DistanceValue, V: Vertex<T>>(&self, props: &[f32]) -> Result<Vec<f32>, String> {
        if props.is_empty() || (props.len() % V::NUM_FEATURES != 0) {
            return Err(format!(
                "Number of features in data ({}) does not match number of features in model ({})",
                props.len(),
                V::NUM_FEATURES
            ));
        }
        let props = props
            .chunks_exact(V::NUM_FEATURES)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();
        let props = DenseMatrix::from_2d_vec(&props).map_err(|e| format!("Failed to create matrix of samples: {e}"))?;

        match self {
            Self::LinearRegression(model) => model
                .predict(&props)
                .map_err(|e| format!("Failed to predict with LinearRegression model: {e}")),
            Self::DecisionTree(model) => model
                .predict(&props)
                .map_err(|e| format!("Failed to predict with DecisionTree model: {e}")),
        }
    }
}
