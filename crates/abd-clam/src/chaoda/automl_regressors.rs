//! Meta-ML regressors built on the [automl] crate

use std::path::Path;

use automl::settings::Algorithm;
use automl::SupervisedModel;

use smartcore::tree::decision_tree_regressor::DecisionTreeRegressorParameters;

use super::metaml::{MetaMLDataset, MetaMLModel};

#[derive(Default)]
/// A metaml wrapper for an [automl] linear regressor
///
pub struct LinearRegressor {
    /// The underlying model for the Linear Regressor.
    #[allow(dead_code)]
    model: Option<SupervisedModel>,
}

impl LinearRegressor {
    /// Constructs a new `LinearRegressor`
    #[must_use]
    pub const fn new() -> Self {
        Self { model: None }
    }
}

impl MetaMLModel for LinearRegressor {
    /// Trains the linear regressor using the provided dataset.
    ///
    /// This function trains the linear regressor using the input `MetaMLDataset` and updates the
    /// model's internal parameters accordingly.
    ///
    /// # Arguments
    ///
    /// * `data`: The dataset used for training the linear regressor.
    ///
    fn train(&mut self, data: MetaMLDataset) {
        let settings = automl::Settings::default_regression().only(Algorithm::Linear);

        let mut model = SupervisedModel::new(data, settings);

        // Train the model
        model.train();

        // Store the model
        self.model = Some(model);
    }
    /// Predict the target value given input features.
    ///
    /// # Arguments
    /// * `features` - The input features for prediction.
    ///
    /// # Returns
    /// The predicted target value.
    fn predict(&self, features: &[f32; 6]) -> Result<f32, String> {
        let Some(model) = self.model.as_ref() else {
            return Err("Model must be trained before being saved".to_string());
        };

        Ok(model.predict(vec![features.to_vec()])[0])
    }
    /// Load a trained model from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file.
    ///
    /// # Returns
    ///
    /// A `Result` containing the loaded model if successful, or an error message if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns an error message if any of the following conditions occur:
    /// * The provided `path` cannot be converted to a string.
    /// * Loading the model from the file fails for any reason.
    fn load(path: &Path) -> Result<Self, String> {
        let Some(path_str) = path.to_str() else {
            return Err("Failed to convert path to a string".to_string());
        };

        let model = SupervisedModel::new_from_file(path_str);

        Ok(Self { model: Some(model) })
    }

    /// Save the trained model to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to save the model to.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success if the model was saved successfully, or an error message if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns an error message if any of the following conditions occur:
    /// * The provided `path` cannot be converted to a string.
    /// * The model has not been trained or is missing when attempting to save it.
    /// * Saving the model to the specified path fails for any reason.
    fn save(&self, path: &Path) -> Result<(), String> {
        let Some(model) = self.model.as_ref() else {
            return Err("Model must be trained before being saved".to_string());
        };

        let Some(path_str) = path.to_str() else {
            return Err("Failed to convert path to a string".to_string());
        };

        model.save(path_str);

        Ok(())
    }
}

#[derive(Default)]
/// A metaml wrapper for an [automl] decision tree regressor
pub struct DecisionTreeRegressor {
    /// The underlying model for the Decision Tree Regressor.
    #[allow(dead_code)]
    model: Option<SupervisedModel>,
}

impl DecisionTreeRegressor {
    /// The maximum depth of the decision tree in a `DecisionTreeRegressor`
    #[allow(dead_code)]
    const MAX_DEPTH: u16 = 3;

    /// Constructs a new `DecisionTreeRegressor` with a specified max tree depth
    #[must_use]
    pub const fn new() -> Self {
        Self { model: None }
    }
}

impl MetaMLModel for DecisionTreeRegressor {
    /// Trains the decision tree regressor using the provided dataset.
    ///
    /// This function trains the decision tree regressor using the input `MetaMLDataset` and updates
    /// the model's internal parameters accordingly.
    ///
    /// # Arguments
    ///
    /// * `data`: The dataset used for training the decision tree regressor.
    ///
    fn train(&mut self, data: MetaMLDataset) {
        let settings = automl::Settings::default_regression()
            .only(Algorithm::DecisionTreeRegressor)
            .with_decision_tree_regressor_settings(
                DecisionTreeRegressorParameters::default().with_max_depth(Self::MAX_DEPTH),
            );

        let mut model = SupervisedModel::new(data, settings);

        // Train the model
        model.train();

        // Store the model
        self.model = Some(model);
    }

    /// Predict the target value given input features.
    ///
    /// # Arguments
    /// * `features` - The input features for prediction.
    ///
    /// # Returns
    /// The predicted target value.
    fn predict(&self, features: &[f32; 6]) -> Result<f32, String> {
        let Some(model) = self.model.as_ref() else {
            return Err("Model must be trained before being saved".to_string());
        };

        Ok(model.predict(vec![features.to_vec()])[0])
    }

    /// Load a trained model from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file.
    ///
    /// # Returns
    ///
    /// A `Result` containing the loaded model if successful, or an error message if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns an error message if any of the following conditions occur:
    /// * The provided `path` cannot be converted to a string.
    /// * Loading the model from the file fails for any reason.
    fn load(path: &Path) -> Result<Self, String> {
        let Some(path_str) = path.to_str() else {
            return Err("Failed to convert path to a string".to_string());
        };

        let model = SupervisedModel::new_from_file(path_str);

        Ok(Self { model: Some(model) })
    }

    /// Save the trained model to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to save the model to.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success if the model was saved successfully, or an error message if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns an error message if any of the following conditions occur:
    /// * The provided `path` cannot be converted to a string.
    /// * The model has not been trained or is missing when attempting to save it.
    /// * Saving the model to the specified path fails for any reason.
    fn save(&self, path: &Path) -> Result<(), String> {
        let Some(model) = &self.model.as_ref() else {
            return Err("Model must be trained before being saved".to_string());
        };

        let Some(path_str) = path.to_str() else {
            return Err("Failed to convert path to a string".to_string());
        };

        model.save(path_str);

        Ok(())
    }
}
