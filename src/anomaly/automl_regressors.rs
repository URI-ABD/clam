//! Meta-ML regressors built on the [automl] crate

use std::path::Path;

use automl::{
    settings::{Algorithm, DecisionTreeRegressorParameters},
    SupervisedModel,
};

use super::metaml::{MetaMLDataset, MetaMLModel};

#[derive(Default)]
/// A metaml wrapper for an [automl] linear regressor
pub struct LinearRegressor {
    model: Option<SupervisedModel>,
}

impl LinearRegressor {
    /// Constructs a new LinearRegressor
    ///
    /// # Examples
    /// ```
    /// # use clam::anomaly::automl_regressors::LinearRegressor;
    /// #
    /// // Construct a new LinearRegressor
    /// let regressor = LinearRegressor::new();
    ///
    /// // Train and use your regressor
    /// ```
    pub fn new() -> Self {
        Self { model: None }
    }
}

impl MetaMLModel for LinearRegressor {
    fn train(&mut self, data: MetaMLDataset) {
        // Create the settings and model
        let settings = automl::Settings::default_regression().only(Algorithm::Linear);
        let mut model = SupervisedModel::new(data, settings);

        // Train the model
        model.train();

        // Store the model
        self.model = Some(model);
    }

    fn predict(&self, features: &[f32; 6]) -> f32 {
        let model = self
            .model
            .as_ref()
            .expect("Model must be trained before making predictions");

        model.predict(vec![features.to_vec()])[0]
    }

    fn load(path: &Path) -> Result<Self, String> {
        let path_str = path.to_str().ok_or("Failed to convert path to a string")?;
        let model = SupervisedModel::new_from_file(path_str);
        Ok(LinearRegressor { model: Some(model) })
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        let model = self.model.as_ref().expect("Model must be trained before being saved.");
        let path_str = path.to_str().ok_or("Failed to convert path to a string")?;
        model.save(path_str);

        Ok(())
    }
}

#[derive(Default)]
/// A metaml wrapper for an [automl] decision tree regressor
pub struct DecisionTreeRegressor {
    model: Option<SupervisedModel>,
}

impl DecisionTreeRegressor {
    /// The maximum depth of the decision tree in a DecisionTreeRegressor
    const MAX_DEPTH: u16 = 3;

    /// Constructs a new DecisionTreeRegressor with a specified max tree depth
    ///
    /// # Examples
    /// ```
    /// # use clam::anomaly::automl_regressors::DecisionTreeRegressor;
    /// #
    /// // Construct a new DecisionTreeRegressor
    /// let regressor = DecisionTreeRegressor::new();
    ///
    /// // Train and use your regressor
    /// ```
    pub fn new() -> Self {
        Self { model: None }
    }
}

impl MetaMLModel for DecisionTreeRegressor {
    fn train(&mut self, data: MetaMLDataset) {
        // Create the settings and model
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

    fn predict(&self, features: &[f32; 6]) -> f32 {
        let model = self
            .model
            .as_ref()
            .expect("Model must be trained before making predictions");

        model.predict(vec![features.to_vec()])[0]
    }

    fn load(path: &Path) -> Result<Self, String> {
        let path_str = path.to_str().ok_or("Failed to convert path to a string")?;
        let model = SupervisedModel::new_from_file(path_str);
        Ok(DecisionTreeRegressor { model: Some(model) })
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        let model = self.model.as_ref().expect("Model must be trained before being saved");
        let path_str = path.to_str().ok_or("Failed to convert path to a string")?;
        model.save(path_str);
        Ok(())
    }
}
