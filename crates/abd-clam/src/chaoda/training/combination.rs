//! Utilities for handling a pair of `MetaMLModel` and `GraphAlgorithm`.

use distances::Number;

use crate::{
    chaoda::{
        inference, roc_auc_score, training::GraphEvaluator, Graph, GraphAlgorithm, TrainableMetaMlModel, Vertex,
        NUM_RATIOS,
    },
    Cluster, Dataset,
};

/// A trainable combination of a `MetaMLModel` and a `GraphAlgorithm`.
#[derive(Clone)]
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
        }
    }

    /// Get the meta-ML scorer function in a callable for any number of `Vertex`es.
    pub fn meta_ml_scorer<I, U, D, S>(&self) -> Result<impl Fn(&[&Vertex<I, U, D, S>]) -> Vec<f32> + '_, String>
    where
        U: Number,
        D: Dataset<I, U>,
        S: Cluster<I, U, D>,
    {
        let model = self.meta_ml.train(NUM_RATIOS, &self.train_x, &self.train_y)?;
        let scorer = move |clusters: &[&Vertex<I, U, D, S>]| {
            let props = clusters.iter().flat_map(|c| c.ratios()).collect::<Vec<_>>();
            model.predict(&props).unwrap_or_else(|e| unreachable!("{e}"))
        };
        Ok(scorer)
    }

    /// Create a `Graph` from the `root` with the given `data` and `min_depth`
    /// using the `TrainedMetaMLModel`.
    pub fn create_graph<'a, I, U, D, S>(
        &self,
        root: &'a Vertex<I, U, D, S>,
        data: &D,
        min_depth: usize,
    ) -> Graph<'a, I, U, D, S>
    where
        U: Number,
        D: Dataset<I, U>,
        S: Cluster<I, U, D>,
    {
        if self.train_x.is_empty() || self.train_y.is_empty() {
            let cluster_scorer = |clusters: &[&Vertex<I, U, D, S>]| {
                clusters
                    .iter()
                    .map(|c| {
                        if c.depth() == min_depth || (c.is_leaf() && c.depth() < min_depth) {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            };
            Graph::from_root(root, data, cluster_scorer, min_depth)
        } else {
            let cluster_scorer = self.meta_ml_scorer().unwrap_or_else(|e| unreachable!("{e}"));
            Graph::from_root(root, data, cluster_scorer, min_depth)
        }
    }

    /// Train the model using the given `Graph`.
    ///
    /// This will first append the anomaly properties and the suitability scores
    /// of the `Cluster`s in the `Graph` to the training data. After that, it
    /// will train the model using the training data. Finally, it will return
    /// the trained model.
    pub fn train_step<I, U, D, S>(
        &mut self,
        graph: &mut Graph<I, U, D, S>,
        labels: &[bool],
    ) -> Result<inference::TrainedCombination, String>
    where
        U: Number,
        D: Dataset<I, U>,
        S: Cluster<I, U, D>,
    {
        let props = graph.iter_anomaly_properties().flatten().copied().collect::<Vec<f32>>();
        self.train_x.extend_from_slice(&props);

        let predictions = self.graph_algorithm.evaluate_points(graph);
        let roc_scores = graph
            .iter_clusters()
            .map(|c| {
                let mut y_true = c.indices().map(|i| labels[i]).collect::<Vec<_>>();
                let mut y_score = c.indices().map(|i| predictions[i]).collect::<Vec<_>>();

                // TODO: This is a temporary fix for the case where there is only one
                // class in the cluster. This should be handled in a better way.
                y_true.push(true);
                y_score.push(1.0);
                y_true.push(false);
                y_score.push(0.0);

                roc_auc_score(&y_true, &y_score)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if roc_scores.len() != props.len() / NUM_RATIOS {
            return Err("The number of clusters does not match the number of properties".to_string());
        }
        self.train_y.extend_from_slice(&roc_scores);

        let meta_ml = self.meta_ml.train(NUM_RATIOS, &self.train_x, &self.train_y)?;
        let graph_algorithm = self.graph_algorithm.clone();

        Ok(inference::TrainedCombination::new(meta_ml, graph_algorithm))
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
        })
    }
}
