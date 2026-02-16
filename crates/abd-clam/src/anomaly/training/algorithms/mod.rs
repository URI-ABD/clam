//! The individual algorithms that make up the CHAODA ensemble.

mod cc;
mod gn;
mod pc;
mod sc;
mod sp;
mod vd;

use crate::{
    chaoda::{Graph, Vertex},
    utils, DistanceValue,
};

/// The algorithms that make up the CHAODA ensemble.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum GraphAlgorithm {
    /// The Cluster Cardinality algorithm.
    CC(cc::ClusterCardinality),
    /// The Graph Neighborhood algorithm.
    GN(gn::GraphNeighborhood),
    /// The Parent Cardinality algorithm.
    PC(pc::ParentCardinality),
    /// The Subgraph Cardinality algorithm.
    SC(sc::SubgraphCardinality),
    /// The Stationary Probability algorithm.
    SP(sp::StationaryProbability),
    /// The Vertex Degree algorithm.
    VD(vd::VertexDegree),
}

impl Default for GraphAlgorithm {
    fn default() -> Self {
        Self::PC(pc::ParentCardinality)
    }
}

impl TryFrom<&str> for GraphAlgorithm {
    type Error = String;

    fn try_from(model: &str) -> Result<Self, Self::Error> {
        Ok(match model {
            "cc" | "CC" | "ClusterCardinality" => Self::CC(cc::ClusterCardinality),
            "gn" | "GN" | "GraphNeighborhood" => Self::GN(gn::GraphNeighborhood::default()),
            "pc" | "PC" | "ParentCardinality" => Self::PC(pc::ParentCardinality),
            "sc" | "SC" | "SubgraphCardinality" => Self::SC(sc::SubgraphCardinality),
            "sp" | "SP" | "StationaryProbability" => Self::SP(sp::StationaryProbability::default()),
            "vd" | "VD" | "VertexDegree" => Self::VD(vd::VertexDegree),
            _ => return Err(format!("Unknown model: {model}")),
        })
    }
}

impl GraphAlgorithm {
    /// Create the default set of algorithms for the CHAODA ensemble.
    #[must_use]
    pub fn default_algorithms() -> Vec<Self> {
        vec![
            Self::CC(cc::ClusterCardinality),
            Self::GN(gn::GraphNeighborhood::default()),
            Self::PC(pc::ParentCardinality),
            Self::SC(sc::SubgraphCardinality),
            Self::SP(sp::StationaryProbability::default()),
            Self::VD(vd::VertexDegree),
        ]
    }

    /// Get the name of the algorithm.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::CC(_) => "CC",
            Self::GN(_) => "GN",
            Self::PC(_) => "PC",
            Self::SC(_) => "SC",
            Self::SP(_) => "SP",
            Self::VD(_) => "VD",
        }
    }
}

impl<T: DistanceValue, V: Vertex<T>> GraphEvaluator<T, V> for GraphAlgorithm {
    fn evaluate_clusters(&self, g: &Graph<T, V>) -> Vec<f32> {
        match self {
            Self::CC(a) => a.evaluate_clusters(g),
            Self::GN(a) => a.evaluate_clusters(g),
            Self::PC(a) => a.evaluate_clusters(g),
            Self::SC(a) => a.evaluate_clusters(g),
            Self::SP(a) => a.evaluate_clusters(g),
            Self::VD(a) => a.evaluate_clusters(g),
        }
    }

    fn normalize_by_cluster(&self) -> bool {
        match self {
            Self::CC(a) => <cc::ClusterCardinality as GraphEvaluator<T, V>>::normalize_by_cluster(a),
            Self::GN(a) => <gn::GraphNeighborhood as GraphEvaluator<T, V>>::normalize_by_cluster(a),
            Self::PC(a) => <pc::ParentCardinality as GraphEvaluator<T, V>>::normalize_by_cluster(a),
            Self::SC(a) => <sc::SubgraphCardinality as GraphEvaluator<T, V>>::normalize_by_cluster(a),
            Self::SP(a) => <sp::StationaryProbability as GraphEvaluator<T, V>>::normalize_by_cluster(a),
            Self::VD(a) => <vd::VertexDegree as GraphEvaluator<T, V>>::normalize_by_cluster(a),
        }
    }
}

/// A trait for how a `Graph` should be evaluated into anomaly scores.
pub trait GraphEvaluator<T: DistanceValue, V: Vertex<T>> {
    /// Evaluate the algorithm on a `Graph` and return a vector of scores for each
    /// `OddBall` in the `Graph`.
    ///
    /// The output vector must be the same length as the number of `OddBall`s in
    /// the `Graph`, and the order of the scores must correspond to the order of the
    /// `OddBall`s in the `Graph`.
    fn evaluate_clusters(&self, g: &Graph<T, V>) -> Vec<f32>;

    /// Whether to normalize anomaly scores by cluster or by point.
    fn normalize_by_cluster(&self) -> bool;

    /// Have points inherit scores from `OddBall`s.
    fn inherit_scores(&self, g: &Graph<T, V>, scores: &[f32]) -> Vec<f32> {
        let mut points_scores = vec![0.0; g.population()];
        for (c, &s) in g.iter_vertices().zip(scores.iter()) {
            for i in c.indices() {
                points_scores[i] = s;
            }
        }
        points_scores
    }

    /// Compute the anomaly scores for all points in the `Graph`.
    ///
    /// This method is a convenience method that wraps the `evaluate` and `inherit_scores`
    /// methods. It evaluates the algorithm on the `Graph` and then inherits the scores
    /// from the `OddBall`s to the points. It correctly handles normalization by cluster
    /// or by point.
    ///
    /// # Returns
    ///
    /// * A vector of anomaly scores for each point in the `Graph`.
    fn evaluate_points(&self, g: &Graph<T, V>) -> Vec<f32> {
        let cluster_scores = {
            let scores = self.evaluate_clusters(g);
            if self.normalize_by_cluster() {
                self.normalize_scores(&scores)
            } else {
                scores
            }
        };

        let scores = self.inherit_scores(g, &cluster_scores);
        if self.normalize_by_cluster() {
            scores
        } else {
            self.normalize_scores(&scores)
        }
    }

    /// Normalize the scores using the Error Function.
    fn normalize_scores(&self, scores: &[f32]) -> Vec<f32> {
        let mean = utils::mean(scores);
        let sd = utils::standard_deviation(scores);
        utils::normalize_1d(scores, mean, sd)
    }
}
