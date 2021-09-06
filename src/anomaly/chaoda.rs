use std::sync::Arc;

use ndarray::prelude::*;
use rayon::prelude::*;

use crate::anomaly;
use crate::criteria;
use crate::prelude::*;

/// CLI Description
/// ex: clam chaoda --(bench/train/infer/...)  <int: max-tree-depth (opt)> <int: min-leaf-size (opt)> <int: speed-threshold (opt)> <Path: out (opt: stdout)> <Path: dataset> <String[]: metrics>
/// In bench mode, output running time of scoring everything to standard error. Do not include IO time to read the dataset.

type NameMethodGraph<T, U> = (
    String,
    Arc<anomaly::IndividualAlgorithm<T, U>>,
    Arc<Graph<T, U>>,
);

pub struct Chaoda<T: Number + 'static, U: Number + 'static> {
    manifolds: Vec<Arc<Manifold<T, U>>>,
    names_methods_graphs: Vec<NameMethodGraph<T, U>>,
    pub scores: Vec<f64>,
}

impl<T: Number + 'static, U: Number + 'static> Chaoda<T, U> {
    pub fn new(
        datasets: Vec<Arc<dyn Dataset<T, U>>>,
        max_tree_depth: Option<usize>,
        min_leaf_size: Option<usize>,
        cluster_scorers: Vec<(&str, Arc<criteria::MetaMLScorer>)>,
        min_selection_depth: Option<usize>,
        use_speed_threshold: bool,
    ) -> Self {
        let mut chaoda = Chaoda {
            manifolds: Vec::new(),
            names_methods_graphs: Vec::new(),
            scores: Vec::new(),
        };
        let min_leaf_size = min_leaf_size.unwrap_or_else(|| {
            std::cmp::max(1, datasets.get(0).unwrap().cardinality() / 1000)
        });
        chaoda.manifolds = chaoda.create_manifolds(
            datasets,
            max_tree_depth.unwrap_or(25),
            min_leaf_size,
        );

        chaoda.names_methods_graphs = chaoda
            .create_graphs(cluster_scorers, min_selection_depth.unwrap_or(4));

        chaoda.scores = chaoda.calculate_anomaly_scores(use_speed_threshold);

        chaoda
    }

    fn create_manifolds(
        &self,
        datasets: Vec<Arc<dyn Dataset<T, U>>>,
        max_tree_depth: usize,
        min_leaf_size: usize,
    ) -> Vec<Arc<Manifold<T, U>>> {
        let partition_criteria = &[
            criteria::max_depth(max_tree_depth),
            criteria::min_cardinality(min_leaf_size),
        ];

        datasets
            .par_iter()
            .map(|dataset| {
                Manifold::new(Arc::clone(dataset), partition_criteria)
            })
            .collect()
    }

    fn create_graphs(
        &self,
        criteria: Vec<(&str, Arc<criteria::MetaMLScorer>)>,
        min_selection_depth: usize,
    ) -> Vec<NameMethodGraph<T, U>> {
        let (meta_ml_names, meta_ml_scorers): (Vec<_>, Vec<_>) =
            criteria.into_iter().unzip();
        let individual_algorithms = anomaly::get_individual_algorithms::<T, U>();

        let graph_algorithms: Vec<_> = meta_ml_names
            .par_iter()
            .map(|&meta_ml_name| {
                individual_algorithms
                    .iter()
                    .filter(|(name, _)| meta_ml_name.contains(*name))
                    .map(|(_, algorithm)| algorithm)
                    .next()
                    .unwrap()
            })
            .collect();

        self.manifolds
            .par_iter()
            .flat_map(|manifold| {
                meta_ml_names
                    .par_iter()
                    .zip(
                        manifold
                            .create_optimal_graphs(
                                &meta_ml_scorers,
                                min_selection_depth,
                            )
                            .into_par_iter(),
                    )
                    .zip(graph_algorithms.par_iter())
                    .map(|((&name, graph), algorithm)| {
                        (name.to_string(), Arc::clone(algorithm), graph)
                    })
            })
            .collect()
    }

    fn calculate_anomaly_scores(&self, use_speed_threshold: bool) -> Vec<f64> {
        let cardinality = self.manifolds.get(0).unwrap().dataset.cardinality();
        let speed_threshold = if use_speed_threshold {
            std::cmp::max(128, (cardinality as f64).sqrt() as usize)
        } else {
            cardinality
        };

        let individual_scores: Vec<_> = self
            .names_methods_graphs
            .par_iter()
            .filter(|(name, _, graph)| {
                name.contains(&graph.metric_name)
                    && (name.contains("cc")
                        || name.contains("cr")
                        || name.contains("vd")
                        || graph.cardinality <= speed_threshold)
            })
            .map(|(_, method, graph)| method(graph))
            .map(Array1::from_vec)
            .filter(|scores| scores.std(0.) > 1e-1)
            .collect();

        if individual_scores.is_empty() {
            vec![0.5; cardinality]
        } else {
            let scores = Array2::from_shape_vec(
                (individual_scores.len(), cardinality),
                individual_scores.into_iter().flatten().collect(),
            )
            .unwrap();

            scores.mean_axis(Axis(0)).unwrap().to_vec()
        }
    }
}
