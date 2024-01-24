//! Criteria used for selecting `Cluster`s for `Graph`s.

/// A `Box`ed function that assigns a score for a given `Cluster`.
pub type MetaMLScorer = Box<fn(crate::core::cluster::Ratios) -> f64>;

use crate::core::graph::_graph::{ClusterSet, EdgeSet};
use crate::{Cluster, Dataset, Edge, Instance};
use distances::Number;
use std::collections::{BinaryHeap, HashSet};

use std::cmp::Ordering;

/// A struct to hold cluster and associated score
pub struct ClusterWrapper<'a, U: Number> {
    /// A cluster that has been scored
    cluster: &'a Cluster<U>,
    /// A score associated to the cluster
    pub score: f64,
}

impl<'a, U: Number> PartialEq for ClusterWrapper<'a, U> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl<'a, U: Number> Eq for ClusterWrapper<'a, U> {}

impl<'a, U: Number> Ord for ClusterWrapper<'a, U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

impl<'a, U: Number> PartialOrd for ClusterWrapper<'a, U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Scores a cluster using the provided scoring function.
///
/// # Arguments
///
/// * `root`: The root of the tree as a `Cluster`.
/// * `scoring_function`: The function used to score each cluster.
///
/// # Returns
///
/// Returns a `Result` containing a `BinaryHeap` of `ClusterWrappers` if successful.
///
/// # Errors
///
/// Returns an `Err` in the following cases:
///
/// * If `ratios()` on a cluster returns `None`, indicating ratios are not available.
///
pub fn score_clusters<'a, U: Number>(
    root: &'a Cluster<U>,
    scoring_function: &crate::core::graph::MetaMLScorer,
) -> Result<BinaryHeap<ClusterWrapper<'a, U>>, String> {
    let mut scored_clusters: BinaryHeap<ClusterWrapper<'a, U>> = BinaryHeap::new();

    for cluster in root.subtree() {
        let score = match cluster.ratios() {
            Some(ratios) => scoring_function(ratios),
            None => return Err("Error: tree must be built with ratios".to_string()),
        };
        scored_clusters.push(ClusterWrapper { cluster, score });
    }

    Ok(scored_clusters)
}

/// Gets optimal clusters to build a graph such that no chosen cluster has an ancestor of descendant in the set
///
/// # Arguments
///
/// * `root` : `Cluster` of the root of a tree
/// * `scoring_function` : a function that scores a given cluster
///
/// # Returns:
///
/// `ClusterSet` of chosen clusters representing highest scored clusters in the tree
///
/// # Errors
///
/// If `ClusterWrapper` contains an invalid cluster-score pairing, or if invalid scoring function name
///
pub fn select_clusters<'a, U: Number>(
    root: &'a Cluster<U>,
    scoring_function: &MetaMLScorer,
) -> Result<ClusterSet<'a, U>, String> {
    let mut scored_clusters = score_clusters(root, scoring_function)?;
    scored_clusters.retain(|item| item.cluster.depth() > 3);

    let mut selected_clusters: HashSet<&'a Cluster<U>> = HashSet::new();

    while !scored_clusters.is_empty() {
        let Some(wrapper) = scored_clusters.pop() else {
            return Err("Invalid ClusterWrapper passed to `get_clusterset`".to_string());
        };
        let best = wrapper.cluster;
        scored_clusters.retain(|item| !item.cluster.is_ancestor_of(best) && !item.cluster.is_descendant_of(best));
        selected_clusters.insert(best);
    }

    Ok(selected_clusters)
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
    clusters: &ClusterSet<'a, U>,
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
