//! Clustered Hierarchical Anomaly and Outlier Detection Algorithms

mod cluster;
mod graph;
mod members;
mod meta_ml;
mod single_metric_model;

use distances::Number;

pub use cluster::Vertex;
pub use graph::Graph;
pub use members::{Algorithm, Member};
pub use meta_ml::MlModel;

use crate::{adapter::Adapter, Cluster, Dataset, Metric, Partition};
use single_metric_model::SingleMetricModel;

/// The combination of a member and meta-ml models for each metric.
type ModelCombinations<const M: usize> = [Vec<(Member, Vec<MlModel>)>; M];

/// Clustered Hierarchical Anomaly and Outlier Detection Algorithms.
///
/// # Type Parameters
///
/// * `I` - The type of the instances.
/// * `U` - The type of the distances between instances.
/// * `D` - The type of the dataset.
/// * `S` - The type of the source `Cluster` for the `Vertex`es.
/// * `M` - The number of metrics.
pub struct Chaoda<'a, I: Clone, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>, const M: usize> {
    /// The individual models for each metric in use.
    #[allow(dead_code)]
    models: [SingleMetricModel<'a, I, U, D, S>; M],
}

impl<'a, I: Clone, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>, const M: usize> Chaoda<'a, I, U, D, S, M> {
    /// Create new trees for use with CHAODA.
    ///
    /// # Arguments
    ///
    /// * `datasets` - The datasets to use for building the trees.
    /// * `criteria` - The criteria for partitioning clusters, one for each dataset.
    /// * `metrics` - The metrics to use for each dataset.
    ///
    /// # Type Parameters
    ///
    /// * `N` - The number of datasets.
    /// * `M` - The number of metrics.
    /// * `C` - The criteria for partitioning clusters.
    ///
    /// # Returns
    ///
    /// For each combination of dataset and metric, the root of the tree.
    #[must_use]
    pub fn new_trees<const N: usize, C: Fn(&S) -> bool>(
        datasets: &mut [D; N],
        criteria: &[C; N],
        metrics: &[Metric<I, U>; M],
        seed: Option<u64>,
    ) -> [[Vertex<I, U, D, S>; M]; N]
    where
        S: Partition<I, U, D>,
    {
        let mut trees = Vec::new();

        for (data, criteria) in datasets.iter_mut().zip(criteria) {
            let mut m_trees = Vec::new();
            for metric in metrics {
                data.set_metric(metric.clone());
                let source = S::new_tree(data, criteria, seed);
                let (vertex, _) = Vertex::adapt_tree(source, None);
                m_trees.push(vertex);
            }
            let m_trees: [Vertex<I, U, D, S>; M] = m_trees
                .try_into()
                .unwrap_or_else(|_| unreachable!("We built a tree for each metric."));
            trees.push(m_trees);
        }

        trees
            .try_into()
            .unwrap_or_else(|_| unreachable!("We built trees for each dataset."))
    }

    /// Returns the default model combinations for CHAODA.
    #[must_use]
    pub fn default_model_combinations() -> ModelCombinations<M> {
        let mut combinations = Vec::new();

        for _ in 0..M {
            let n_combinations = Member::default_members()
                .into_iter()
                .map(|member| (member, MlModel::default_models()))
                .collect();
            combinations.push(n_combinations);
        }

        combinations
            .try_into()
            .unwrap_or_else(|_| unreachable!("We built model combinations for each N."))
    }

    /// Create a new CHAODA instance.
    ///
    /// # Arguments
    ///
    /// * `metrics` - The metrics to use in the ensemble.
    /// * `model_combinations` - The model combinations to use for each metric.
    /// * `min_depth` - The minimum depth of the `Vertex`es to consider.
    #[must_use]
    pub fn new(metrics: [Metric<I, U>; M], model_combinations: Option<ModelCombinations<M>>, min_depth: usize) -> Self {
        let model_combinations = model_combinations.unwrap_or_else(|| Self::default_model_combinations());

        let models = metrics
            .into_iter()
            .zip(model_combinations)
            .map(|(metric, model_combinations)| SingleMetricModel::new(metric, model_combinations, min_depth))
            .collect::<Vec<_>>();

        let models = models
            .try_into()
            .unwrap_or_else(|_| unreachable!("We built a model for each metric."));

        Self { models }
    }
}
