//! Utilities for handling a pair of `MetaMLModel` and `GraphAlgorithm`.

use crate::{
    chaoda::{inference, roc_auc_score, training::GraphEvaluator, Graph, GraphAlgorithm, TrainableMetaMlModel, Vertex},
    DistanceValue,
};

/// A trainable combination of a `MetaMLModel` and a `GraphAlgorithm`.
#[derive(Clone, Default)]
#[allow(clippy::module_name_repetitions)]
pub struct TrainableCombination {
    /// The `MetaMLModel` to use.
    pub meta_ml: TrainableMetaMlModel,
    /// The `GraphAlgorithm` to use.
    pub graph_algorithm: GraphAlgorithm,
    /// The data used to train the model so far.
    /// The rows are the anomaly properties of the `Cluster`s from the `Graph`s.
    /// The elements are stored in row-major order, assuming 6 columns.
    pub train_x: Vec<f32>,
    /// The labels used to train the model so far. Each is the suitability of
    /// the `Cluster`s for selection in the `Graph`.
    pub train_y: Vec<f32>,
    /// The ROC AUC score of the most recent training step.
    pub roc_score: f32,
}

impl TrainableCombination {
    /// Get the name of the combination.
    ///
    /// The name is in the format `{meta_ml.short_name()}-{graph_algorithm.name()}`.
    pub fn name(&self) -> String {
        format!("{}-{}", self.meta_ml.short_name(), self.graph_algorithm.name())
    }

    /// Create a new `TrainableCombination` with the given `MetaMLModel` and `GraphAlgorithm`.
    #[must_use]
    pub const fn new(meta_ml: TrainableMetaMlModel, graph_algorithm: GraphAlgorithm) -> Self {
        Self {
            meta_ml,
            graph_algorithm,
            train_x: Vec::new(),
            train_y: Vec::new(),
            roc_score: 0.5,
        }
    }

    /// Create training data from the given `Graph`.
    ///
    /// # Arguments
    ///
    /// - `graph`: The `Graph` to use.
    /// - `labels`: The labels to use.
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - An array of:
    ///   - the flattened anomaly properties of the `Cluster`s.
    ///   - the suitability scores of the `Cluster`s for `Graph` selection.
    /// - The ROC AUC score of the `Graph`.
    ///
    /// # Errors
    ///
    /// - If any roc-auc score calculation fails.
    pub fn data_from_graph<T: DistanceValue, V: Vertex<T>>(
        &self,
        graph: &Graph<T, V>,
        labels: &[bool],
    ) -> Result<([Vec<f32>; 2], f32), String> {
        let props = graph
            .iter_anomaly_properties()
            .flat_map(|v| v.as_ref().to_vec())
            .collect::<Vec<f32>>();
        let predictions = self.graph_algorithm.evaluate_points(graph);

        let scores = graph
            .iter_vertices()
            .map(|c| {
                // Get the labels and predictions and append a dummy true and false value to avoid empty classes for roc_auc_score
                let indices = c.indices();
                let y_true = indices
                    .iter()
                    .map(|&i| labels[i])
                    .chain(std::iter::once(true))
                    .chain(std::iter::once(false))
                    .collect::<Vec<_>>();
                let y_pred = indices
                    .iter()
                    .map(|&i| predictions[i])
                    .chain(std::iter::once(1.0))
                    .chain(std::iter::once(0.0))
                    .collect::<Vec<_>>();

                let roc_score = roc_auc_score(&y_true, &y_pred)?;
                // Use (2 * |roc_score - 0.5|) as the prediction target.
                // This is to steer the training towards assigning high scores
                // to `Cluster`s whose ROC AUC score is very different from 0.5.
                // `Cluster`s with high scores will be preferentially selected
                // for `Graph`s.
                let diff = (roc_score - 0.5).abs() * 2.0;
                Ok::<_, String>(diff)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let roc_score = roc_auc_score(labels, &predictions)?;

        Ok(([props, scores], roc_score))
    }

    /// Append the given data to the training data for this model.
    ///
    /// # Arguments
    ///
    /// - `x`: The flattened anomaly properties of the `Cluster`s.
    /// - `y`: The suitability scores of the `Cluster`s for `Graph` selection.
    ///
    /// # Errors
    ///
    /// - If the data are empty.
    /// - If the number of properties is not a multiple of `NUM_RATIOS`.
    /// - If the number of samples does not match the number of targets.
    pub fn append_data(&mut self, x: &[f32], y: &[f32], roc_score: Option<f32>) -> Result<(), String> {
        if x.is_empty() || y.is_empty() {
            return Err("The data are empty".to_string());
        }

        self.train_x.extend_from_slice(x);
        self.train_y.extend_from_slice(y);
        self.roc_score = roc_score.unwrap_or(0.5);

        Ok(())
    }

    /// Train the model using the given `Graph`.
    ///
    /// This will first append the anomaly properties and the suitability scores
    /// of the `Cluster`s in the `Graph` to the training data. After that, it
    /// will train the model using the training data. Finally, it will return
    /// the trained model.
    pub fn train_step<T: DistanceValue, V: Vertex<T>>(&self) -> Result<inference::TrainedCombination, String> {
        let meta_ml = self.meta_ml.train(V::NUM_FEATURES, &self.train_x, &self.train_y)?;
        let graph_algorithm = self.graph_algorithm.clone();
        Ok(inference::TrainedCombination::new(
            meta_ml,
            graph_algorithm,
            self.roc_score,
        ))
    }
}

impl TryFrom<&str> for TrainableCombination {
    type Error = String;

    fn try_from(name: &str) -> Result<Self, Self::Error> {
        let parts: Vec<&str> = name.split('-').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid name: {name}"));
        }
        let [meta_ml, graph_algorithm] = [parts[0], parts[1]];
        let meta_ml = TrainableMetaMlModel::try_from(meta_ml)?;
        let graph_algorithm = GraphAlgorithm::try_from(graph_algorithm)?;
        Ok(Self {
            meta_ml,
            graph_algorithm,
            train_x: Vec::new(),
            train_y: Vec::new(),
            roc_score: 0.5,
        })
    }
}
