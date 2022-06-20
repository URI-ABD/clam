use std::collections::HashMap;
use std::hash::Hash;

use crate::prelude::*;
use crate::utils::helpers;

pub type ClusterScores<'a, T, U> = HashMap<&'a Cluster<'a, T, U>, f64>;
pub type InstanceScores = HashMap<usize, f64>;

pub trait GraphScorer: Hash {
    fn call<'a, T: Number, U: Number>(&self, graph: &'a Graph<T, U>) -> (ClusterScores<'a, T, U>, Vec<f64>) {
        let cluster_scores = {
            let mut cluster_scores = self.score_graph(graph);
            if self.normalize_on_clusters() {
                let (clusters, scores): (Vec<_>, Vec<_>) = cluster_scores.into_iter().unzip();
                cluster_scores = clusters
                    .into_iter()
                    .zip(helpers::normalize_1d(&scores).into_iter())
                    .collect();
            }
            cluster_scores
        };

        let instance_scores = {
            let mut instance_scores = self.inherit_scores(&cluster_scores);
            if !self.normalize_on_clusters() {
                let (indices, scores): (Vec<_>, Vec<_>) = instance_scores.into_iter().unzip();
                instance_scores = indices
                    .into_iter()
                    .zip(helpers::normalize_1d(&scores).into_iter())
                    .collect();
            }
            instance_scores
        };

        let scores_array = self.ordered_scores(&instance_scores);

        (cluster_scores, scores_array)
    }

    fn name(&self) -> &str;

    fn short_name(&self) -> &str;

    fn normalize_on_clusters(&self) -> bool;

    fn score_graph<'a, T: Number, U: Number>(&self, graph: &'a Graph<T, U>) -> ClusterScores<'a, T, U>;

    fn inherit_scores<T: Number, U: Number>(&self, scores: &ClusterScores<T, U>) -> InstanceScores {
        scores
            .iter()
            .flat_map(|(&c, &s)| c.indices().into_iter().map(move |i| (i, s)))
            .collect()
    }

    fn ordered_scores(&self, scores: &InstanceScores) -> Vec<f64> {
        let mut scores: Vec<_> = scores.iter().map(|(&i, &s)| (i, s)).collect();
        scores.sort_by_key(|(i, _)| *i);
        let (_, scores): (Vec<_>, Vec<f64>) = scores.into_iter().unzip();
        scores
    }
}

pub struct ClusterCardinality;

impl Hash for ClusterCardinality {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        GraphScorer::name(self).hash(state)
    }
}

impl GraphScorer for ClusterCardinality {
    fn name(&self) -> &str {
        "cluster_cardinality"
    }

    fn short_name(&self) -> &str {
        "cc"
    }

    fn normalize_on_clusters(&self) -> bool {
        true
    }

    fn score_graph<'a, T: Number, U: Number>(&self, graph: &'a Graph<T, U>) -> ClusterScores<'a, T, U> {
        graph
            .ordered_clusters()
            .iter()
            .map(|&c| (c, c.cardinality() as f64))
            .collect()
    }
}
