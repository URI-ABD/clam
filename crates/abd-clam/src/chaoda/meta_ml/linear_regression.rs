//! Meta-ML with a linear regression model.

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression as sc_lr;

use crate::Cluster;

use super::{super::AnomalyFeatures, MetaMlPredictor, MetaMlTrainer};

/// A linear regression model.
#[must_use]
#[derive(Debug)]
pub struct LinearRegression {
    /// The underlying model from smartcore.
    model: sc_lr::LinearRegression<f64, f64, DenseMatrix<f64>, Vec<f64>>,
}

impl From<LinearRegression> for super::MetaMlModel {
    fn from(alg: LinearRegression) -> Self {
        Self::LinearRegression(alg)
    }
}

impl MetaMlPredictor for LinearRegression {
    fn predict<T, A>(&self, cluster: &Cluster<T, (A, AnomalyFeatures)>) -> Result<f64, String> {
        let features = cluster.annotation.1.as_vec();
        let features = DenseMatrix::from_2d_vec(&vec![features]).map_err(|e| e.to_string())?;
        let prediction = self.model.predict(&features).map_err(|e| e.to_string())?;
        Ok(prediction[0])
    }
}

impl MetaMlTrainer for LinearRegression {
    fn fit(data: &[(AnomalyFeatures, f64)]) -> Result<Self, String> {
        let (features, targets): (Vec<_>, Vec<_>) = data.iter().map(|(x, y)| (x.as_vec(), *y)).unzip();
        let features = DenseMatrix::from_2d_vec(&features).map_err(|e| e.to_string())?;

        let params = sc_lr::LinearRegressionParameters::default();
        let model = sc_lr::LinearRegression::fit(&features, &targets, params).map_err(|e| e.to_string())?;

        Ok(Self { model })
    }
}
