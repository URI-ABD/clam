use crate::chaoda::pretrained_models;
use crate::{Cluster, ClusterSet};
use distances::Number;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

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

/// Scores a cluster with a given scoring fuction
///
///
/// # Arguments
///
/// * `root`: The root of the tree.
/// * `scoring_function`: Function in which to score each cluster
///
/// # Returns:
///
/// `BinaryHeap` of `ClusterWrappers`
///
#[must_use]
pub fn score_clusters<'a, U: Number>(
    root: &'a Cluster<U>,
    scoring_function: &crate::core::graph::MetaMLScorer,
) -> BinaryHeap<ClusterWrapper<'a, U>> {
    let clusters = root.subtree();
    let mut scored_clusters: BinaryHeap<ClusterWrapper<'a, U>> = BinaryHeap::new();

    for cluster in clusters {
        let score = cluster.ratios().map_or(0.0, scoring_function);
        scored_clusters.push(ClusterWrapper { cluster, score });
    }

    scored_clusters
}

/// Gets scoring function from name
///
/// # Arguments
///
/// * `input_string` : `String` of the name of the function
/// * `functions` : `Vec` of scoring functions with associated names
///
/// # Returns:
///
/// Selected scoring function
///
fn get_function_by_name<'a>(
    input_string: &str,
    functions: &'a Vec<(&'a str, crate::core::graph::MetaMLScorer)>,
) -> Option<crate::core::graph::MetaMLScorer> {
    for (name, function) in functions {
        if name == &input_string {
            return Some(function.clone());
        }
    }
    None
}

/// Gets `ClusterSet` from `root` and `scoring_function`
///
/// # Arguments
///
/// * `root` : `Cluster` of the root of a tree
/// * `scoring_function` :
///
/// # Returns:
///
/// `ClusterSet` of chosen clusters representing highest scored with no ancestors or descendants
///
/// # Errors
///
/// If `ClusterWrapper` contains an invalid cluster-score pairing, or if invalid scoring function name
///
pub fn select_optimal_graph_clusters<'a, U: Number>(
    root: &'a Cluster<U>,
    scoring_function: &'a str,
) -> Result<ClusterSet<'a, U>, String> {
    let scorers = pretrained_models::get_meta_ml_scorers();

    return get_function_by_name(scoring_function, &scorers).map_or_else(
        || Err(format!("Scoring function {scoring_function} not found")),
        |scoring_function| {
            let scored_clusters = score_clusters(root, &scoring_function);
            select_optimal_clusters(scored_clusters)
        },
    );
}

/// Gets `ClusterSet` from `BinaryHeap` of `ClusterWrappers`
///
/// # Arguments
///
/// * clusters : `BinaryHeap` of `ClusterWrappers` containing a cluster and its score
///
/// # Returns:
///
/// `ClusterSet` of chosen clusters representing highest scored with no ancestors or descendants
///
/// # Errors
///
/// If `ClusterWrapper` contains an invalid cluster-score pairing
///
pub fn select_optimal_clusters<'a, U: Number>(
    mut clusters: BinaryHeap<ClusterWrapper<'a, U>>,
) -> Result<ClusterSet<'a, U>, String> {
    let mut cluster_set: HashSet<&'a Cluster<U>> = HashSet::new();

    while !clusters.is_empty() {
        let Some(wrapper) = clusters.pop() else {
            return Err("Invalid ClusterWrapper passed to `get_clusterset`".to_string());
        };
        let best = wrapper.cluster;
        clusters = clusters
            .into_iter()
            .filter(|item| !item.cluster.is_ancestor_of(best) && !item.cluster.is_descendant_of(best))
            .collect();
        cluster_set.insert(best);
    }

    Ok(cluster_set)
}
