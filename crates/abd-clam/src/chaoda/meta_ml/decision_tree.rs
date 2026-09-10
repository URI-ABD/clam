//! Meta-ML with a decision tree regression model.

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_regressor as sc_dt;

use crate::Cluster;

use super::{super::AnomalyFeatures, MetaMlPredictor, MetaMlTrainer};

/// A decision tree model.
#[must_use]
#[derive(Debug)]
pub struct DecisionTree {
    /// The underlying decision tree model.
    model: sc_dt::DecisionTreeRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>,
}

impl From<DecisionTree> for super::MetaMlModel {
    fn from(alg: DecisionTree) -> Self {
        Self::DecisionTree(alg)
    }
}

impl MetaMlPredictor for DecisionTree {
    fn predict<T, A>(&self, cluster: &Cluster<T, (A, AnomalyFeatures)>) -> Result<f64, String> {
        let features = cluster.annotation.1.as_vec();
        let features = DenseMatrix::from_2d_vec(&vec![features]).map_err(|e| e.to_string())?;
        let prediction = self.model.predict(&features).map_err(|e| e.to_string())?;
        Ok(prediction[0])
    }
}

impl MetaMlTrainer for DecisionTree {
    fn fit(data: &[(AnomalyFeatures, f64)]) -> Result<Self, String> {
        let (features, targets): (Vec<_>, Vec<_>) = data.iter().map(|(x, y)| (x.as_vec(), *y)).unzip();
        let features = DenseMatrix::from_2d_vec(&features).map_err(|e| e.to_string())?;

        let params = sc_dt::DecisionTreeRegressorParameters::default();
        let model = sc_dt::DecisionTreeRegressor::fit(&features, &targets, params).map_err(|e| e.to_string())?;

        Ok(Self { model })
    }
}
