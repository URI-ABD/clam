use crate::chaoda::pretrained_models;
use crate::core::cluster::Ratios;
use crate::{Cluster, ClusterSet};
use distances::Number;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

pub struct ClusterWrapper<'a, U: Number> {
    cluster: &'a Cluster<U>,
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

// pub fn avg_score(ratio: Ratios) -> f64 {
//     let mut score: f64 = 0.0;
//     let mut count: f64 = 0.0;
//     let scorers = pretrained_models::get_meta_ml_scorers();
//     for model in scorers {
//         score += model.1(ratio);
//         count += 1.0;
//     }
//     return score / count;
// }

fn score_clusters<'a, U: Number>(
    root: &'a Cluster<U>,
    scoring_function: crate::core::graph::MetaMLScorer,
) -> BinaryHeap<ClusterWrapper<'a, U>> {
    let mut clusters = root.subtree();
    let mut scored_clusters: BinaryHeap<ClusterWrapper<'a, U>> = BinaryHeap::new();

    for cluster in clusters {
        let cluster_score = cluster.ratios().map_or(0.0, |value| scoring_function(value));
        scored_clusters.push(ClusterWrapper {
            cluster,
            score: cluster_score,
        });
    }

    return scored_clusters;
}

fn select_optimal_clusters<'a, U: Number>(clusters: BinaryHeap<ClusterWrapper<'a, U>>) -> ClusterSet<'a, U> {
    let mut cluster_set: HashSet<&'a Cluster<U>> = HashSet::new();
    let mut clusters: BinaryHeap<&ClusterWrapper<'a, U>> = BinaryHeap::from(clusters.iter().collect::<Vec<_>>());

    clusters.retain(|item| item.cluster.depth() > 3);

    loop {
        if clusters.len() == 0 {
            break;
        }
        let best = clusters.pop().unwrap().cluster;
        clusters = clusters
            .into_iter()
            .filter(|item| !item.cluster.is_ancestor_of(best) && !item.cluster.is_descendant_of(best))
            .collect();
        cluster_set.insert(best);
    }

    cluster_set
}

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

pub fn select_optimal_graph_clusters<U: Number>(
    root: &Cluster<U>,
    scoring_function: String,
) -> Result<ClusterSet<U>, String> {
    let scorers = pretrained_models::get_meta_ml_scorers();

    return get_function_by_name(scoring_function.as_str(), &scorers).map_or_else(
        || Err(format!("Scoring function {scoring_function} not found")),
        |scoring_function| {
            let scored_clusters = score_clusters(root, scoring_function);
            Ok(select_optimal_clusters(scored_clusters))
        },
    );
}
