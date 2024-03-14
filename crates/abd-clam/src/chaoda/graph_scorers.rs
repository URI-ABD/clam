//! Different ways to score clustered graphs based on their properties.

use std::collections::HashMap;
use std::hash::Hash;

use super::graph::Graph;
use super::Vertex;
use crate::utils::{mean, standard_deviation};
use distances::Number;

use crate::Cluster;

/// Type alias for cluster scores associated with clusters in a graph.
pub type ClusterScores<'a, U> = HashMap<&'a Vertex<U>, f64>;
/// Type alias for scores associated with individual instances or elements.
pub type InstanceScores = HashMap<usize, f64>;

/// A trait for scoring graphs.
pub trait GraphScorer<'a, U: Number>: Hash {
    /// Computes scores for the given graph and returns cluster scores and an array of scores.
    ///
    /// This function is responsible for calculating scores based on the input graph.
    ///
    /// # Arguments
    ///
    /// * `_graph`: A reference to the input graph from which scores are calculated.
    ///
    /// # Returns
    ///
    /// A tuple containing cluster scores and an array of scores.
    ///
    /// * `ClusterScores`: A structure that holds scores associated with individual clusters in the graph.
    /// * `Vec<f64>`: An array of scores, where each element corresponds to a specific aspect or metric.
    ///
    /// # Errors
    ///
    /// Throws an error if unable to compute scores for the given graph
    ///
    fn call(&self, graph: &'a Graph<'a, U>) -> Result<(ClusterScores<'a, U>, Vec<f64>), String> {
        let cluster_scores = {
            let scores = self.score_graph(graph)?;
            let mut cluster_scores: ClusterScores<'a, U> = scores;
            if self.normalize_on_clusters() {
                let (clusters, scores): (Vec<_>, Vec<_>) = cluster_scores.into_iter().unzip();
                cluster_scores = clusters
                    .into_iter()
                    .zip(crate::utils::normalize_1d(
                        &scores,
                        mean(&scores),
                        standard_deviation(&scores),
                    ))
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
                    .zip(crate::utils::normalize_1d(
                        &scores,
                        mean(&scores),
                        standard_deviation(&scores),
                    ))
                    .collect();
            }
            instance_scores
        };

        let scores_array = self.ordered_scores(&instance_scores);

        Ok((cluster_scores, scores_array))
    }

    /// Returns the name of the graph scorer.
    fn name(&self) -> &str;

    /// Returns the short name of the graph scorer.
    fn short_name(&self) -> &str;

    /// Indicates whether normalization should be performed based on clusters.
    fn normalize_on_clusters(&self) -> bool;

    /// Computes and returns cluster scores for clusters in the input graph.
    ///
    /// This function calculates cluster scores based on the characteristics of clusters in the input graph.
    /// Cluster scores are represented as a `ClusterScores` mapping clusters to their associated scores.
    ///
    /// # Arguments
    ///
    /// * `graph`: The input graph for which cluster scores are computed.
    ///
    /// # Returns
    ///
    /// A `ClusterScores` mapping clusters in the input graph to their associated scores.
    ///
    /// # Errors
    ///
    /// Throws an error if unable to compute a score for a cluster within a given graph.
    ///
    fn score_graph(&self, graph: &'a Graph<'a, U>) -> Result<ClusterScores<'a, U>, String>;

    /// Inherits cluster scores and computes scores for individual instances.
    ///
    /// This function inherits cluster scores and uses them to calculate scores for individual instances.
    /// It takes a `ClusterScores` as input, which maps clusters to their associated scores,
    /// and returns an `InstanceScores` mapping instances to their computed scores.
    ///
    /// # Arguments
    ///
    /// * `scores`: A `ClusterScores` mapping clusters to their associated scores.
    ///
    /// # Returns
    ///
    /// An `InstanceScores` mapping instances to their computed scores.
    fn inherit_scores(&self, scores: &ClusterScores<U>) -> InstanceScores {
        scores
            .iter()
            .flat_map(|(&c, &s)| c.indices().map(move |i| (i, s)))
            .collect()
    }

    /// Orders the scores for individual instances.
    ///
    /// This function takes an `InstanceScores` mapping instances to their associated scores
    /// and returns a sorted vector of scores for instances. The vector contains the scores
    /// in ascending order based on the instance indices.
    ///
    /// # Arguments
    ///
    /// * `scores`: An `InstanceScores` mapping instances to their associated scores.
    ///
    /// # Returns
    ///
    /// A sorted vector of scores for instances in ascending order.
    fn ordered_scores(&self, scores: &InstanceScores) -> Vec<f64> {
        let mut scores: Vec<_> = scores.iter().map(|(&i, &s)| (i, s)).collect();
        scores.sort_by_key(|(i, _)| *i);
        let (_, scores): (Vec<_>, Vec<f64>) = scores.into_iter().unzip();
        scores
    }
}

/// A graph scorer that calculates scores based on cluster cardinality.
pub struct ClusterCardinality;

impl Hash for ClusterCardinality {
    /// Generates a hash for the `ClusterCardinality` instance.
    ///
    /// This function hashes the string "`cluster_cardinality`" to uniquely identify this scorer.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "cluster_cardinality".hash(state);
    }
}

impl<'a, U: Number> GraphScorer<'a, U> for ClusterCardinality {
    /// Returns the name of the `ClusterCardinality` graph scorer.
    ///
    /// The name is "`cluster_cardinality`."
    fn name(&self) -> &str {
        "cluster_cardinality"
    }

    /// Returns the short name of the `ClusterCardinality` graph scorer.
    ///
    /// The short name is "cc."
    fn short_name(&self) -> &str {
        "cc"
    }

    /// Indicates whether normalization should be performed based on clusters for `ClusterCardinality`.
    fn normalize_on_clusters(&self) -> bool {
        true
    }

    /// Computes and returns cluster scores based on cluster cardinality.
    ///
    /// This function calculates the scores for clusters based on their cardinality and returns them
    /// as a map of cluster references to their respective scores as floating-point values. The scores
    /// are calculated based on the cluster's cardinality, which represents the number of elements
    /// in the cluster.
    ///
    /// # Arguments
    ///
    /// * `_graph`: A reference to the input graph from which cluster scores are calculated.
    ///
    /// # Returns
    ///
    /// A `ClusterScores` mapping clusters to their calculated scores based on their cardinality.
    fn score_graph(&self, graph: &'a Graph<'a, U>) -> Result<ClusterScores<'a, U>, String> {
        let scores = graph
            .ordered_clusters()
            .iter()
            .map(|&c| (c, -c.cardinality().as_f64()))
            .collect();
        Ok(scores)
    }
}

/// A graph scorer that calculates scores based on component cardinality.
pub struct ComponentCardinality;

impl Hash for ComponentCardinality {
    /// Generates a hash for the `ComponentCardinality` instance.
    ///
    /// This function hashes the string "`component_cardinality`" to uniquely identify this scorer.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "component_cardinality".hash(state);
    }
}

impl<'a, U: Number> GraphScorer<'a, U> for ComponentCardinality {
    /// Returns the name of the `ComponentCardinality` graph scorer.
    ///
    /// The name is "`component_cardinality`."
    fn name(&self) -> &str {
        "component_cardinality"
    }

    /// Returns the short name of the `ComponentCardinality` graph scorer.
    ///
    /// The short name is "sc."
    fn short_name(&self) -> &str {
        "sc"
    }

    /// Indicates whether normalization should be performed based on clusters for `ComponentCardinality`.
    fn normalize_on_clusters(&self) -> bool {
        true
    }

    /// Computes and returns cluster scores based on component cardinality.
    ///
    /// This function calculates the scores for clusters based on the cardinality of components they belong to
    /// and returns them as a map of cluster references to their respective scores as floating-point values.
    /// The scores are calculated based on the cardinality of the components that each cluster belongs to.
    ///
    /// # Arguments
    ///
    /// * `_graph`: A reference to the input graph from which cluster scores are calculated.
    ///
    /// # Returns
    ///
    /// A `ClusterScores` mapping clusters to their calculated scores based on the cardinality of their components.
    fn score_graph(&self, graph: &'a Graph<'a, U>) -> Result<ClusterScores<'a, U>, String> {
        let scores = graph
            .find_component_clusters()
            .iter()
            .flat_map(|clusters| {
                let score = -clusters.len().as_f64();
                clusters.iter().map(move |&c| (c, score))
            })
            .collect();
        Ok(scores)
    }
}

/// A graph scorer that calculates scores based on vertex degree.
pub struct VertexDegree;

impl Hash for VertexDegree {
    /// Generates a hash for the `VertexDegree` instance.
    ///
    /// This function hashes the string "`vertex_degree`" to uniquely identify this scorer.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "vertex_degree".hash(state);
    }
}

impl<'a, U: Number> GraphScorer<'a, U> for VertexDegree {
    /// Returns the name of the `VertexDegree` graph scorer.
    ///
    /// The name is "`vertex_degree`."
    fn name(&self) -> &str {
        "vertex_degree"
    }

    /// Returns the short name of the `VertexDegree` graph scorer.
    ///
    /// The short name is "vd."
    fn short_name(&self) -> &str {
        "vd"
    }

    /// Indicates whether normalization should be performed based on clusters for `VertexDegree`.
    fn normalize_on_clusters(&self) -> bool {
        true
    }

    /// Computes and returns cluster scores based on vertex degree.
    ///
    /// This function calculates the scores for clusters based on the vertex degrees of their vertices
    /// and returns them as a map of cluster references to their respective scores as floating-point values.
    /// The scores are calculated based on the vertex degrees of the clusters' vertices within the input graph.
    ///
    /// # Arguments
    ///
    /// * `_graph`: A reference to the input graph from which cluster scores are calculated.
    ///
    /// # Returns
    ///
    /// A `ClusterScores` mapping clusters to their calculated scores based on the vertex degrees of their vertices.
    #[allow(clippy::cast_precision_loss)]
    fn score_graph(&self, graph: &'a Graph<'a, U>) -> Result<ClusterScores<'a, U>, String> {
        let scores: Result<ClusterScores<'a, U>, String> = graph
            .ordered_clusters()
            .iter()
            .map(|&c| graph.vertex_degree(c).map(|degree| (c, -(degree as f64))))
            .collect();
        scores
    }
}

/// A graph scorer that calculates scores based on parent-child cluster relationships and cardinality.
///
/// This scorer assigns scores to clusters based on the cardinality of a cluster relative to its parent
/// cluster in a hierarchical structure. It uses a user-defined weight function to assign weights to
/// different levels of the hierarchy.
pub struct ParentCardinality<'a, U: Number> {
    /// The root cluster in the hierarchical structure.
    #[allow(dead_code)]
    root: &'a Vertex<U>,
    /// User-defined weight function for hierarchy levels.
    #[allow(dead_code)]
    weight: Box<dyn (Fn(usize) -> f64) + Send + Sync>,
}

impl<'a, U: Number> ParentCardinality<'a, U> {
    /// Creates a new instance of the `ParentCardinality` scorer.
    ///
    /// The `root` parameter specifies the root cluster of the hierarchy. The weight function is used
    /// to assign weights to different levels of the hierarchy.
    ///
    /// # Arguments
    ///
    /// * `_root`: The root cluster of the hierarchical structure.
    ///
    /// # Returns
    ///
    /// A new instance of the `ParentCardinality` scorer with the specified root cluster and weight function.
    pub fn new(root: &'a Vertex<U>) -> Self {
        let weight = Box::new(|d: usize| 1. / (d.as_f64()).sqrt());
        Self { root, weight }
    }

    /// Computes the ancestry of a given cluster.
    ///
    /// The ancestry of a cluster is a list of clusters starting from the root and ending at the given
    /// cluster. The list represents the hierarchical relationship between clusters in the structure.
    ///
    /// # Arguments
    ///
    /// * `_c`: The cluster for which the ancestry is to be computed.
    ///
    /// # Returns
    ///
    /// A vector of references to clusters representing the ancestry of the specified cluster.
    ///
    /// This method computes the ancestry of a cluster by traversing the hierarchical structure, starting from
    /// the root cluster and following parent-child relationships until the given cluster is reached.
    /// The resulting vector contains references to clusters that form the ancestry of the specified cluster.
    pub fn ancestry(&self, _c: &'a Vertex<U>) -> Vec<&'a Vertex<U>> {
        todo!()

        // c.history().into_iter().fold(vec![self.root], |mut ancestors, turn| {
        //     let last = ancestors.last().unwrap();
        //     let [left, right] = last.children().unwrap();
        //     let child = if *turn { right } else { left };
        //     ancestors.push(child);
        //     ancestors
        // })
    }
}

impl<'a, U: Number> Hash for ParentCardinality<'a, U> {
    /// Generates a hash for the `ParentCardinality` instance.
    ///
    /// This function hashes the string "`parent_cardinality`" to uniquely identify this scorer.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "parent_cardinality".hash(state);
    }
}

impl<'a, U: Number> GraphScorer<'a, U> for ParentCardinality<'a, U> {
    /// Returns the name of the `ParentCardinality` graph scorer.
    ///
    /// The name is "`parent_cardinality`."
    fn name(&self) -> &str {
        "parent_cardinality"
    }

    /// Returns the short name of the `ParentCardinality` graph scorer.
    ///
    /// The short name is "pc."
    fn short_name(&self) -> &str {
        "pc"
    }

    /// Indicates whether normalization should be performed based on clusters for `ParentCardinality`.
    ///
    /// TODO!
    fn normalize_on_clusters(&self) -> bool {
        todo!()
        //true
    }

    /// Computes and returns cluster scores based on parent-child cluster relationships and cardinality.
    ///
    /// This function calculates the scores for clusters based on their hierarchical relationships and
    /// cardinality, using the weight function to assign weights to different hierarchy levels.
    ///
    /// # Arguments
    ///
    /// * `_graph`: The input graph for which cluster scores are to be computed.
    ///
    /// # Returns
    ///
    /// A map of cluster indices to their respective scores as floating-point values.
    fn score_graph(&self, graph: &'a Graph<'a, U>) -> Result<ClusterScores<'a, U>, String> {
        let scores = graph
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
                        (self.weight)(i + 1) * parent.cardinality().as_f64() / child.cardinality().as_f64()
                    })
                    .sum();
                (c, -score)
            })
            .collect();
        Ok(scores)
    }
}

/// A graph scorer that calculates scores based on the neighborhood of clusters in a graph.
pub struct GraphNeighborhood {
    /// Fraction used to determine neighborhood size.
    #[allow(dead_code)]
    eccentricity_fraction: f64,
}

impl GraphNeighborhood {
    /// Creates a new instance of the `GraphNeighborhood` scorer.
    ///
    /// The `eccentricity_fraction` parameter specifies a factor that influences the number of steps taken
    /// in the neighborhood computation.
    ///
    #[must_use]
    pub const fn new(eccentricity_fraction: f64) -> Self {
        Self { eccentricity_fraction }
    }

    /// Calculates the number of steps for neighborhood computation in the graph.
    ///
    /// This function computes the number of steps based on the eccentricity of the given cluster
    /// and the eccentricity fraction specified during initialization.
    ///
    /// # Arguments
    ///
    /// * `_graph`: The input graph for which the number of steps is calculated.
    /// * `_c`: The cluster for which the number of steps is determined.
    ///
    /// # Returns
    ///
    /// The number of steps for neighborhood computation as a `usize` value.
    #[allow(dead_code)]
    fn num_steps<'a, U: Number>(&self, _graph: &'a Graph<'a, U>, _c: &'a Vertex<U>) -> usize {
        todo!()

        // let steps = graph.unchecked_eccentricity(c) as f64 * self.eccentricity_fraction;
        // 1 + steps as usize
    }
}

impl Hash for GraphNeighborhood {
    /// Generates a hash for the `GraphNeighborhood` instance.
    ///
    /// This function hashes the string "`graph_neighborhood`" to uniquely identify this scorer.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "graph_neighborhood".hash(state);
    }
}

impl<'a, U: Number> GraphScorer<'a, U> for GraphNeighborhood {
    /// Returns the name of the `GraphNeighborhood` graph scorer.
    ///
    /// The name is "`graph_neighborhood`."
    fn name(&self) -> &str {
        "graph_neighborhood"
    }

    /// Returns the short name of the `GraphNeighborhood` graph scorer.
    ///
    /// The short name is "gn."
    fn short_name(&self) -> &str {
        "gn"
    }

    /// Indicates whether normalization should be performed based on clusters for `GraphNeighborhood`.
    ///
    /// TODO!
    fn normalize_on_clusters(&self) -> bool {
        todo!()
        //true
    }

    /// Computes and returns cluster scores based on the neighborhood of clusters in the graph.
    ///
    /// This function calculates the scores for clusters based on their neighborhood in the graph,
    /// considering the number of steps and the size of the clusters within the neighborhood.
    ///
    /// # Arguments
    ///
    /// * `_graph`: The input graph for which cluster scores are to be computed.
    ///
    /// # Returns
    ///
    /// A map of cluster indices to their respective scores as floating-point values.
    fn score_graph(&self, _graph: &'a Graph<'a, U>) -> Result<ClusterScores<'a, U>, String> {
        todo!()

        // graph
        //     .ordered_clusters()
        //     .iter()
        //     .map(|&c| {
        //         let steps = self.num_steps(graph, c);
        //         // TODO: Do we need +1?
        //         let score = (0..steps + 1)
        //             .zip(graph.unchecked_frontier_sizes(c).iter())
        //             .fold(0, |score, (_, &size)| score + size);
        //         (c, -(score as f64))
        //     })
        //     .collect()
    }
}

/// A graph scorer that calculates stationary probabilities of clusters after a specified number of steps.
pub struct StationaryProbabilities {
    /// Number of steps for stationary probability calculation.
    #[allow(dead_code)]
    num_steps: usize,
}

impl Hash for StationaryProbabilities {
    /// Generates a hash for the `StationaryProbabilities` instance.
    ///
    /// This function hashes the string "`stationary_probabilities`" to uniquely identify this scorer.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "stationary_probabilities".hash(state);
    }
}

impl StationaryProbabilities {
    /// Creates a new instance of the `StationaryProbabilities` scorer.
    ///
    /// The `num_steps` parameter specifies the number of steps for which stationary probabilities will be computed.
    ///
    #[must_use]
    pub const fn new(num_steps: usize) -> Self {
        Self { num_steps }
    }
}

impl<'a, U: Number> GraphScorer<'a, U> for StationaryProbabilities {
    /// Returns the name of the `StationaryProbabilities` graph scorer.
    ///
    /// The name is "`stationary_probabilities`."
    fn name(&self) -> &str {
        "stationary_probabilities"
    }

    /// Returns the short name of the `StationaryProbabilities` graph scorer.
    ///
    /// The short name is "sp."
    fn short_name(&self) -> &str {
        "sp"
    }

    /// Indicates whether normalization should be performed based on clusters for `StationaryProbabilities`.
    ///
    /// TODO!
    fn normalize_on_clusters(&self) -> bool {
        todo!()
        //true
    }

    #[allow(unused_variables)]

    /// Computes and returns cluster scores based on stationary probabilities after a specified number of steps.
    ///
    /// This function calculates the scores for clusters based on their stationary probabilities in the graph
    /// after a fixed number of steps. The `num_steps` parameter specified during initialization determines
    /// the number of steps.
    ///
    /// # Arguments
    ///
    /// * `graph`: The input graph for which cluster scores are to be computed.
    ///
    /// # Returns
    ///
    /// A map of cluster indices to their respective scores as floating-point values.
    fn score_graph(&self, graph: &'a Graph<U>) -> Result<ClusterScores<'a, U>, String> {
        todo!()
    }
}
