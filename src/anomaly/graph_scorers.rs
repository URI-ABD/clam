use std::collections::HashMap;
use std::hash::Hash;

use crate::prelude::*;
use crate::utils::helpers;

pub type ClusterScores<'a, T> = HashMap<&'a Cluster<'a, T>, f64>;
pub type InstanceScores = HashMap<usize, f64>;

pub trait GraphScorer<'a, T: Number>: Hash {
    fn call(&self, graph: &'a Graph<'a, T>) -> (ClusterScores<'a, T>, Vec<f64>) {
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

    fn score_graph(&self, graph: &'a Graph<'a, T>) -> ClusterScores<'a, T>;

    fn inherit_scores(&self, scores: &ClusterScores<T>) -> InstanceScores {
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
        "cluster_cardinality".hash(state)
    }
}

impl<'a, T: Number> GraphScorer<'a, T> for ClusterCardinality {
    fn name(&self) -> &str {
        "cluster_cardinality"
    }

    fn short_name(&self) -> &str {
        "cc"
    }

    fn normalize_on_clusters(&self) -> bool {
        true
    }

    fn score_graph(&self, graph: &'a Graph<'a, T>) -> ClusterScores<'a, T> {
        graph
            .ordered_clusters()
            .iter()
            .map(|&c| (c, c.cardinality() as f64))
            .collect()
    }
}

pub struct ComponentCardinality;

impl Hash for ComponentCardinality {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "component_cardinality".hash(state)
    }
}

impl<'a, T: Number> GraphScorer<'a, T> for ComponentCardinality {
    fn name(&self) -> &str {
        "component_cardinality"
    }

    fn short_name(&self) -> &str {
        "sc"
    }

    fn normalize_on_clusters(&self) -> bool {
        true
    }

    fn score_graph(&self, graph: &'a Graph<'a, T>) -> ClusterScores<'a, T> {
        graph
            .find_component_clusters()
            .iter()
            .flat_map(|clusters| {
                let score = clusters.len() as f64;
                clusters.iter().map(move |&c| (c, score))
            })
            .collect()
    }
}

pub struct VertexDegree;

impl Hash for VertexDegree {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "vertex_degree".hash(state)
    }
}

impl<'a, T: Number> GraphScorer<'a, T> for VertexDegree {
    fn name(&self) -> &str {
        "vertex_degree"
    }

    fn short_name(&self) -> &str {
        "vd"
    }

    fn normalize_on_clusters(&self) -> bool {
        true
    }

    fn score_graph(&self, graph: &'a Graph<'a, T>) -> ClusterScores<'a, T> {
        graph
            .ordered_clusters()
            .iter()
            .map(|&c| (c, graph.unchecked_vertex_degree(c) as f64))
            .collect()
    }
}

pub struct ParentCardinality<'a, T: Number> {
    root: &'a Cluster<'a, T>,
    weight: Box<dyn (Fn(usize) -> f64) + Send + Sync>,
}

impl<'a, T: Number> ParentCardinality<'a, T> {
    pub fn new(root: &'a Cluster<'a, T>) -> Self {
        let weight = Box::new(|d: usize| 1. / (d as f64).sqrt());
        Self { root, weight }
    }

    pub fn ancestry(&self, c: &'a Cluster<'a, T>) -> Vec<&'a Cluster<'a, T>> {
        c.name()
            .iter()
            .map(|b| *b)
            .fold(vec![self.root], |mut ancestors, turn| {
                let last = ancestors.last().unwrap();
                let child = if turn { last.right_child() } else { last.left_child() };
                ancestors.push(child);
                ancestors
            })
    }
}

impl<'a, T: Number> Hash for ParentCardinality<'a, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "parent_cardinality".hash(state)
    }
}

impl<'a, T: Number> GraphScorer<'a, T> for ParentCardinality<'a, T> {
    fn name(&self) -> &str {
        "parent_cardinality"
    }

    fn short_name(&self) -> &str {
        "pc"
    }

    fn normalize_on_clusters(&self) -> bool {
        true
    }

    fn score_graph(&self, graph: &'a Graph<'a, T>) -> ClusterScores<'a, T> {
        graph
            .ordered_clusters()
            .iter()
            .map(|&c| {
                let ancestry = self.ancestry(c);
                let score: f64 = ancestry
                    .iter()
                    .skip(1)
                    .zip(ancestry.iter())
                    .enumerate()
                    .map(|(i, (child, parent))| {
                        (self.weight)(i + 1) * parent.cardinality() as f64 / child.cardinality() as f64
                    })
                    .sum();
                (c, -score)
            })
            .collect()
    }
}

pub struct GraphNeighborhood {
    eccentricity_fraction: f64,
}

impl GraphNeighborhood {
    pub fn new(eccentricity_fraction: f64) -> Self {
        Self { eccentricity_fraction }
    }

    fn num_steps<'a, T: Number>(&self, graph: &'a Graph<'a, T>, c: &'a Cluster<'a, T>) -> usize {
        let steps = graph.unchecked_eccentricity(c) as f64 * self.eccentricity_fraction;
        1 + steps as usize
    }
}

impl Hash for GraphNeighborhood {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "graph_neighborhood".hash(state)
    }
}

impl<'a, T: Number> GraphScorer<'a, T> for GraphNeighborhood {
    fn name(&self) -> &str {
        "graph_neighborhood"
    }

    fn short_name(&self) -> &str {
        "gn"
    }

    fn normalize_on_clusters(&self) -> bool {
        true
    }

    fn score_graph(&self, graph: &'a Graph<'a, T>) -> ClusterScores<'a, T> {
        graph
            .ordered_clusters()
            .iter()
            .map(|&c| {
                let steps = self.num_steps(graph, c);
                // TODO: Do we need +1?
                let score = (0..steps + 1)
                    .zip(graph.unchecked_frontier_sizes(c).iter())
                    .fold(0, |score, (_, &size)| score + size);
                (c, -(score as f64))
            })
            .collect()
    }
}

#[allow(dead_code)]
pub struct StationaryProbabilities {
    num_steps: usize,
}

impl Hash for StationaryProbabilities {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "stationary_probabilities".hash(state)
    }
}

impl StationaryProbabilities {
    pub fn new(num_steps: usize) -> Self {
        Self { num_steps }
    }
}

impl<'a, T: Number> GraphScorer<'a, T> for StationaryProbabilities {
    fn name(&self) -> &str {
        "stationary_probabilities"
    }

    fn short_name(&self) -> &str {
        "sp"
    }

    fn normalize_on_clusters(&self) -> bool {
        true
    }

    #[allow(unused_variables)]
    fn score_graph(&self, graph: &'a Graph<T>) -> ClusterScores<'a, T> {
        todo!()
    }
}
