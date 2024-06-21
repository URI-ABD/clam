//! Criteria used for selecting `Cluster`s for `Graph`s.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use distances::Number;

use super::{Edge, EdgeSet, Ratios, Vertex, VertexSet};
use crate::{Cluster, Dataset, Instance};

/// A `Box`ed function that assigns a score for a given `Cluster`.
pub type MetaMLScorer = Box<fn(Ratios) -> f64>;

/// A Wrapper that contains a cluster and its score
struct VertexWrapper<'a, U: Number> {
    /// A cluster
    pub cluster: &'a Vertex<U>,
    /// An associated score
    pub score: f64,
}

impl<'a, U: Number> PartialEq for VertexWrapper<'a, U> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl<'a, U: Number> Eq for VertexWrapper<'a, U> {}

// impl<'a, U: Number> Ord for ClusterWrapper<'a, U> {
//     fn cmp(&self, other: &Self) -> Ordering {
//         self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
//     }
// }

impl<'a, U: Number> Ord for VertexWrapper<'a, U> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal) {
            Ordering::Equal => {
                // If scores are equal, use a tiebreaker comparison based on offset - lower offset wins
                match self.cluster.offset().cmp(&other.cluster.offset()) {
                    Ordering::Equal => {
                        // If offsets are equal, use a tiebreaker comparison based on cardinality
                        self.cluster.cardinality().cmp(&other.cluster.cardinality())
                    }
                    ord => ord.reverse(),
                }
            }
            ord => ord,
        }
    }
}

impl<'a, U: Number> PartialOrd for VertexWrapper<'a, U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Scores a cluster with a given scoring function
///
/// # Arguments
///
/// * `root`: The root of the tree.
/// * `scoring_function`: Function in which to score each cluster
///
/// # Returns:
///
/// `BinaryHeap` of `ClusterWrappers`
fn score_clusters<'a, U: Number>(
    root: &'a Vertex<U>,
    scoring_function: &super::MetaMLScorer,
) -> BinaryHeap<VertexWrapper<'a, U>> {
    let mut scored_clusters: BinaryHeap<VertexWrapper<'a, U>> = BinaryHeap::new();

    for cluster in root.subtree() {
        let score = scoring_function(cluster.ratios());
        scored_clusters.push(VertexWrapper { cluster, score });
    }

    scored_clusters
}

/// Gets `ClusterSet` from `BinaryHeap` of `ClusterWrappers`
///
/// # Arguments
///
/// * `clusters` : `BinaryHeap` of `ClusterWrappers` containing a cluster and its score
/// * `scoring_functions` : `MetaMLScorer` to score the given clusters
/// * `min_depth` : `usize` of the minimum depth to start selecting clusters
///
/// # Returns:
///
/// `ClusterSet` of chosen clusters representing highest scored with no ancestors or descendants
///
/// # Errors
///
/// If `ClusterWrapper` contains an invalid cluster-score pairing
///
pub fn select_clusters<'a, U: Number>(
    root: &'a Vertex<U>,
    scoring_function: &MetaMLScorer,
    min_depth: usize,
) -> Result<VertexSet<'a, U>, String> {
    let mut cluster_set: HashSet<&'a Vertex<U>> = HashSet::new();
    let mut scored_clusters = score_clusters(root, scoring_function);
    scored_clusters.retain(|item| item.cluster.depth() >= min_depth || item.cluster.is_leaf());
    while !scored_clusters.is_empty() {
        let Some(wrapper) = scored_clusters.pop() else {
            return Err("Invalid ClusterWrapper passed to `get_clusterset`".to_string());
        };
        let best = wrapper.cluster;
        scored_clusters.retain(|item| !item.cluster.is_ancestor_of(best) && !item.cluster.is_descendant_of(best));
        cluster_set.insert(best);
    }

    Ok(cluster_set)
}

/// Detects edges between clusters based on their spatial relationships.
///
/// This function iterates through each pair of clusters in the provided `ClusterSet` and
/// checks whether there is an edge between them. An edge is formed if the distance between
/// the centers of two clusters is less than or equal to the sum of their radii.
///
/// # Arguments
///
/// * `clusters`: A reference to a `ClusterSet` containing the clusters to be analyzed.
/// * `data`: A reference to the dataset used to calculate distances between clusters.
///
/// # Returns
///
/// A `HashSet` containing the detected edges, represented by `Edge` instances.
#[allow(clippy::implicit_hasher)]
pub fn detect_edges<'a, I: Instance, U: Number, D: Dataset<I, U>>(
    clusters: &VertexSet<'a, U>,
    data: &D,
) -> EdgeSet<'a, U> {
    // TODO! Refactor for better performance
    // TODO! Generalize over different hashers?...
    let mut edges = HashSet::new();
    for (i, c1) in clusters.iter().enumerate() {
        for (j, c2) in clusters.iter().enumerate().skip(i + 1) {
            if i != j {
                let distance = c1.distance_to_other(data, c2);
                if distance <= c1.radius() + c2.radius() {
                    edges.insert(Edge::new(c1, c2, distance));
                }
            }
        }
    }

    edges
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::{Cluster, PartitionCriteria, Tree, VecDataset};
    use distances::number::Float;
    use distances::Number;
    use rand::SeedableRng;

    use super::*;
    use crate::chaoda::pretrained_models;

    pub fn gen_dataset(
        cardinality: usize,
        dimensionality: usize,
        seed: u64,
        metric: fn(&Vec<f32>, &Vec<f32>) -> f32,
    ) -> VecDataset<Vec<f32>, f32, usize> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let data = symagen::random_data::random_tabular(cardinality, dimensionality, -1., 1., &mut rng);
        let name = "test".to_string();
        VecDataset::new(name, data, metric, false)
    }

    #[allow(clippy::ptr_arg)]
    pub fn euclidean<T: Number, F: Float>(x: &Vec<T>, y: &Vec<T>) -> F {
        distances::vectors::euclidean(x, y)
    }

    #[test]
    fn scoring() {
        let data = gen_dataset(1000, 10, 42, euclidean);

        let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
        let raw_tree = Tree::new(data, Some(42))
            .partition(&partition_criteria, Some(42))
            .normalize_ratios();

        let root = raw_tree.root();

        let mut priority_queue = score_clusters(root, &pretrained_models::get_meta_ml_scorers()[0].1);

        assert_eq!(priority_queue.len(), root.subtree().len());

        let mut prev_value: f64;
        let mut curr_value: f64;

        prev_value = priority_queue.pop().unwrap().score;
        while !priority_queue.is_empty() {
            curr_value = priority_queue.pop().unwrap().score;
            assert!(prev_value >= curr_value);
            prev_value = curr_value;
        }

        let cluster_set = select_clusters(root, &pretrained_models::get_meta_ml_scorers()[0].1, 4).unwrap();
        for i in &cluster_set {
            for j in &cluster_set {
                if i != j {
                    assert!(!i.is_descendant_of(j) && !i.is_ancestor_of(j));
                }
            }
        }

        for i in &root.subtree() {
            let mut ancestor_of = false;
            let mut descendant_of = false;
            for j in &cluster_set {
                if i.is_ancestor_of(j) {
                    ancestor_of = true;
                }
                if i.is_descendant_of(j) {
                    descendant_of = true;
                }
            }
            assert!(ancestor_of || descendant_of || cluster_set.contains(i));
        }
    }
}
