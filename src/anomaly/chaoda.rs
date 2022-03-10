use std::sync::Arc;

use ndarray::prelude::*;

use crate::prelude::*;

use super::MetaML;

/// CLI Description
/// ex: clam chaoda --(bench/train/infer/...)  <int: max-tree-depth (opt)> <int: min-leaf-size (opt)> <int: speed-threshold (opt)> <Path: out (opt: stdout)> <Path: dataset> <String[]: metrics>
/// In bench mode, output running time of scoring everything to standard error. Do not include IO time to read the dataset.

type MmlGraph<T, U> = (MetaML<T, U>, Arc<Graph<T, U>>);

pub struct Chaoda<T: Number + 'static, U: Number + 'static> {
    manifolds: Vec<Arc<Manifold<T, U>>>,
    mml_graphs: Vec<MmlGraph<T, U>>,
    pub scores: Vec<f64>,
}

impl<T: Number + 'static, U: Number + 'static> Chaoda<T, U> {
    pub fn new(
        datasets: Vec<Arc<dyn Dataset<T, U>>>,
        max_tree_depth: Option<usize>,
        min_leaf_size: Option<usize>,
        cluster_scorers: Vec<crate::anomaly::MetaML<T, U>>,
        min_selection_depth: Option<usize>,
        use_speed_threshold: bool,
    ) -> Self {
        let mut chaoda = Chaoda {
            manifolds: Vec::new(),
            mml_graphs: Vec::new(),
            scores: Vec::new(),
        };
        let min_cardinality = datasets.get(0).unwrap().cardinality() / 1000;
        let min_leaf_size = min_leaf_size.unwrap_or_else(|| std::cmp::max(1, min_cardinality));
        chaoda.manifolds = chaoda.create_manifolds(datasets, max_tree_depth.unwrap_or(50), min_leaf_size);
        chaoda.mml_graphs = chaoda.create_graphs(cluster_scorers, min_selection_depth.unwrap_or(4));
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
            .iter()
            .inspect(|&dataset| println!("Building manifold with metric {}", dataset.metric_name()))
            .map(|dataset| Manifold::new(Arc::clone(dataset), partition_criteria))
            .collect()
    }

    #[allow(clippy::needless_collect)]
    fn create_graphs(
        &self,
        mml_methods: Vec<crate::anomaly::MetaML<T, U>>,
        min_selection_depth: usize,
    ) -> Vec<MmlGraph<T, U>> {
        let graphs = mml_methods
            .iter()
            .map(|mml| {
                let manifold = self
                    .manifolds
                    .iter()
                    .find(|&manifold| manifold.metric_name() == mml.metric)
                    .unwrap()
                    .clone();
                let clusters = manifold.select_clusters(&mml.mml_method, min_selection_depth);
                manifold.create_graph(&clusters)
            })
            .map(Arc::new)
            .collect::<Vec<_>>();
        println!("Got {} Graphs.", graphs.len());
        mml_methods.into_iter().zip(graphs.into_iter()).collect()
    }

    fn calculate_anomaly_scores(&self, use_speed_threshold: bool) -> Vec<f64> {
        let cardinality = self.manifolds.get(0).unwrap().dataset.cardinality();
        let speed_threshold = if use_speed_threshold {
            std::cmp::max(128, (cardinality as f64).sqrt() as usize)
        } else {
            cardinality
        };

        let individual_scores = self
            .mml_graphs
            .iter()
            .filter(|(mml, graph)| {
                mml.algorithm_name == "cc"
                    || mml.algorithm_name == "cr"
                    || mml.algorithm_name == "vd"
                    || graph.cardinality <= speed_threshold
            })
            .inspect(|(mml, _)| println!("Applying {} method", mml))
            .map(|(mml, graph)| Arc::clone(&mml.algorithm)(Arc::clone(graph)))
            // .inspect(|scores| self.print_vec(scores))
            .collect::<Vec<_>>();
        println!("Scored {} algorithms.", individual_scores.len());

        if individual_scores.is_empty() {
            vec![0.5; cardinality]
        } else {
            let scores = Array2::from_shape_vec(
                (individual_scores.len(), cardinality),
                individual_scores.into_iter().flatten().collect(),
            )
            .unwrap();

            // let scores = scores.mean_axis(Axis(0)).unwrap().to_vec();
            // self.print_vec(&scores);
            // scores
            scores.mean_axis(Axis(0)).unwrap().to_vec()
        }
    }

    #[allow(dead_code)]
    fn print_vec(&self, values: &[f64]) {
        println!(
            "{:?}\n",
            values
                .iter()
                .map(|v| format!("{:.2}", v))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}
