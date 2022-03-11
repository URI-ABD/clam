use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use bitvec::prelude::*;
use num_traits::FromPrimitive;
use ordered_float::OrderedFloat;
use rayon::prelude::*;

use crate::prelude::*;
use criteria::MetaMLScorer;
use criteria::PartitionCriterion;

/// A `HashMap` from `Clusters` to a `HashMap` of their candidate neighbors and the distance to that candidate.
pub type Candidates<T, U> = HashMap<Arc<Cluster<T, U>>, HashMap<Arc<Cluster<T, U>>, U>>;

/// Cluster ratios of the geometric and topological properties used for anomaly detection.
pub type Ratios = [f64; 6];

type ScoresHeap<T, U> = BinaryHeap<(OrderedFloat<f64>, Arc<Cluster<T, U>>)>;

/// A `Manifold` connects the tree and graph structures and enables higher-order applications such as search, compression, anomaly detection, etc.
#[derive(Debug)]
pub struct Manifold<T: Number, U: Number> {
    pub dataset: Arc<dyn Dataset<T, U>>,
    pub root: Arc<Cluster<T, U>>,
    candidates: Candidates<T, U>,
}

impl<T: Number, U: Number> Manifold<T, U> {
    /// Create a new `Manifold` from a dataset and cluster-partitioning criteria.
    pub fn new(dataset: Arc<dyn Dataset<T, U>>, partition_criteria: &[PartitionCriterion<T, U>]) -> Arc<Self> {
        let mut manifold = Manifold {
            dataset: Arc::clone(&dataset),
            root: Cluster::new_root(dataset).partition(partition_criteria),
            candidates: HashMap::new(),
        };
        manifold.candidates = manifold.compute_candidates();
        Arc::new(manifold)
    }

    /// Returns the name of the metric used in the dataset.
    pub fn metric_name(&self) -> String {
        self.dataset.metric_name()
    }

    /// Returns the number of instances in the dataset.
    pub fn dataset_cardinality(&self) -> usize {
        self.dataset.cardinality()
    }

    /// Computes the candidate neighbors for all clusters in the tree. This is useful for graph construction.
    fn compute_candidates(&self) -> Candidates<T, U> {
        let mut candidates: Candidates<T, U> = [(
            Arc::clone(&self.root),
            [(Arc::clone(&self.root), U::zero())].iter().cloned().collect(),
        )]
        .iter()
        .cloned()
        .collect();

        let mut parents = vec![Arc::clone(&self.root)];
        while !parents.is_empty() {
            let child_candidates: Candidates<T, U> = parents
                .par_iter()
                .flat_map(|parent| match parent.children.read().unwrap().clone() {
                    Some((left, right)) => {
                        let parent_candidates = candidates.get(parent).unwrap();
                        vec![
                            (Arc::clone(&left), left.find_candidates(parent_candidates)),
                            (Arc::clone(&right), right.find_candidates(parent_candidates)),
                        ]
                    }
                    None => Vec::new(),
                })
                .collect();

            candidates.extend(child_candidates.into_iter());

            parents = parents
                .par_iter()
                .flat_map(|parent| match parent.children.read().unwrap().clone() {
                    Some((left, right)) => {
                        vec![left, right]
                    }
                    None => Vec::new(),
                })
                .collect();
        }

        candidates
    }

    /// Returns a `Graph` created from the given clusters.
    pub fn create_graph(&self, clusters: &[Arc<Cluster<T, U>>]) -> Graph<T, U> {
        let edges = clusters
            .par_iter()
            .flat_map(|cluster| {
                let candidates = self.candidates.get(cluster).unwrap();
                clusters
                    .par_iter()
                    .filter(|&candidate| candidates.contains_key(candidate))
                    .map(|neighbor| {
                        Edge::new(
                            Arc::clone(cluster),
                            Arc::clone(neighbor),
                            *candidates.get(neighbor).unwrap(),
                        )
                    })
                    .collect::<HashSet<_>>()
            })
            .collect();

        Graph::new(clusters.iter().cloned().collect(), edges)
    }

    /// Creates all layer-graphs for the manifold.
    /// 
    /// A layer graph contains clusters from a uniform tree-depth (and any leaves that exist at shallower depths) 
    pub fn create_layer_graphs(&self) -> Vec<Arc<Graph<T, U>>> {
        let mut tree = self.root.flatten_tree();
        let mut depth = 0;
        let mut leaves = Vec::new();
        let mut layers = Vec::new();

        while !tree.is_empty() {
            let (mut layer, rest): (Vec<_>, Vec<_>) = tree.into_par_iter().partition(|cluster| cluster.depth() == depth);
            tree = rest;
            depth += 1;

            let mut new_leaves = layer
                .par_iter()
                .filter(|&cluster| cluster.children.read().unwrap().clone().is_none())
                .cloned()
                .collect();
            layer.extend(leaves.iter().cloned());
            leaves.append(&mut new_leaves);

            layers.push(Arc::new(self.create_graph(&layer)));
        }

        layers
    }

    /// Returns clusters selected by some meta-ml criterion. This is used for anomaly detection.
    pub fn select_clusters(&self, criterion: &MetaMLScorer, min_selection_depth: usize) -> Vec<Arc<Cluster<T, U>>> {
        let (shallow_clusters, clusters): (Vec<_>, Vec<_>) = self
            .root
            .flatten_tree()
            .into_iter()
            .partition(|cluster| cluster.depth() < min_selection_depth);

        let mut scores_heap: ScoresHeap<T, U> = clusters
            .into_iter()
            .chain(
                shallow_clusters
                    .into_iter()
                    .filter(|cluster| cluster.children.read().unwrap().is_none()),
            )
            .map(|cluster| (OrderedFloat::from_f64(criterion(cluster.ratios)).unwrap(), cluster))
            .collect();

        let mut selected = Vec::new();
        while !scores_heap.is_empty() {
            let (_, cluster) = scores_heap.pop().unwrap();

            scores_heap = scores_heap
                .into_par_iter()
                .filter(|(_, other)| !other.is_ancestor_of(&cluster) && !cluster.is_ancestor_of(other))
                .collect();

            selected.push(cluster);
        }
        selected
    }

    /// Returns the graph created with clsuters selected by some meta-ml criterion.
    pub fn select_graph(&self, meta_ml: &Arc<MetaMLScorer>, min_selection_depth: usize) -> Arc<Graph<T, U>> {
        let clusters = self.select_clusters(meta_ml, min_selection_depth);
        Arc::new(self.create_graph(&clusters))
    }

    /// Returns the graphs created by a collection of meta-ml criteria.
    pub fn create_optimal_graphs(
        &self,
        meta_ml_scorers: &[Arc<MetaMLScorer>],
        min_selection_depth: usize,
    ) -> Vec<Arc<Graph<T, U>>> {
        meta_ml_scorers
            .iter()
            .map(|criterion| self.select_graph(criterion, min_selection_depth))
            .collect()
    }

    /// For the given cluster name, returns a Vec containing the ancestors of that cluster.
    /// If the cluster is not found in the tree, this returns an Err.
    pub fn ancestry(&self, name: &BitVec) -> Result<Vec<Arc<Cluster<T, U>>>, String> {
        let mut ancestors = vec![Arc::clone(&self.root)];

        for go_right in name.iter().by_ref() {
            let (left, right) = match ancestors.last().unwrap().children.read().unwrap().clone() {
                Some((left, right)) => (left, right),
                None => return Err(format!("cluster {:?} not found in the tree", name)),
            };
            if *go_right {
                ancestors.push(right);
            } else {
                ancestors.push(left);
            }
        }

        Ok(ancestors)
    }

    /// Returns a cluster given the name of that cluster.
    pub fn select(&self, name: BitVec) -> Result<Arc<Cluster<T, U>>, String> {
        Ok(self.ancestry(&name)?.swap_remove(name.len() - 1))
    }
}
