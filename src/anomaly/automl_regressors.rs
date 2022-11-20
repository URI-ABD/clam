//! Meta-ML regressors built on the [automl] crate

use std::path::Path;

use automl::{
    settings::{Algorithm, DecisionTreeRegressorParameters},
    SupervisedModel,
};
use smartcore::dataset::Dataset;

use super::metaml::{MetaMLDataset, MetaMLModel};

// Transforms a MetaMLDataset into an automl dataset
fn metaml_dataset_to_automl(data: MetaMLDataset) -> Dataset<f32, f32> {
    // Double-check meta-information about the data
    // These are invariants in MetaMLDataset, but it can't hurt to be sure ;)
    assert!(data.features.is_empty() || data.features[0].len() == 6);
    assert!(data.features.len() == data.targets.len());

    // Data labels
    let feature_names = vec![
        "Cardinality".to_string(),
        "Radius".to_string(),
        "Local fractal dimension".to_string(),
        "Cardinality (exponential moving average)".to_string(),
        "Radius (exponential moving average)".to_string(),
        "Local fractal dimension (exponential moving average)".to_string(),
    ];
    let target_names = vec!["Expected AUC".to_string()];
    let description = "A dataset for training a meta-ml model to predict ROC AUC based on six inputs".to_string();

    // Construct and return a dataset
    Dataset {
        num_samples: data.features.len(),
        num_features: 6,
        feature_names,
        target_names,
        description,
        data: data.features.into_iter().flatten().collect(),
        target: data.targets,
    }
}

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
        // Construct a metaml dataset from the input data
        let dataset = metaml_dataset_to_automl(data);

        // Create the settings and model
        let settings = automl::Settings::default_regression().only(Algorithm::Linear);
        let mut model = SupervisedModel::new(dataset, settings);

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
        // Construct a metaml dataset from the input data
        let dataset = metaml_dataset_to_automl(data);

        // Create the settings and model
        let settings = automl::Settings::default_regression()
            .only(Algorithm::DecisionTreeRegressor)
            .with_decision_tree_regressor_settings(
                DecisionTreeRegressorParameters::default().with_max_depth(Self::MAX_DEPTH),
            );
        let mut model = SupervisedModel::new(dataset, settings);

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
