//! The individual algorithms that make up the CHAODA ensemble.

mod cc;
mod gn;
mod pc;
mod sc;
mod sp;
mod vd;

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::utils;

use super::Graph;

/// The algorithms that make up the CHAODA ensemble.
#[derive(Clone, Serialize, Deserialize)]
pub enum Member {
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

impl Member {
    /// Create a new `ChaodaMember` algorithm using default parameters.
    ///
    /// # Parameters
    ///
    /// * `model`: The name of the algorithm.
    ///
    /// # Errors
    ///
    /// If the algorithm name is not recognized.
    pub fn new(model: &str) -> Result<Self, String> {
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

    /// Create the default set of algorithms for the CHAODA ensemble.
    #[must_use]
    pub fn default_members() -> Vec<Self> {
        vec![
            Self::CC(cc::ClusterCardinality),
            Self::GN(gn::GraphNeighborhood::default()),
            Self::PC(pc::ParentCardinality),
            Self::SC(sc::SubgraphCardinality),
            // Self::SP(sp::StationaryProbability::default()),
            Self::VD(vd::VertexDegree),
        ]
    }

    /// Get the name of the algorithm.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::CC(a) => a.name(),
            Self::GN(a) => a.name(),
            Self::PC(a) => a.name(),
            Self::SC(a) => a.name(),
            Self::SP(a) => a.name(),
            Self::VD(a) => a.name(),
        }
    }

    /// Evaluate the algorithm on a `Graph` and return a vector of scores for each
    /// `OddBall` in the `Graph`.
    ///
    /// The output vector must be the same length as the number of `OddBall`s in
    /// the `Graph`, and the order of the scores must correspond to the order of the
    /// `OddBall`s in the `Graph`.
    pub fn evaluate_clusters<U: Number>(&self, g: &mut Graph<U>) -> Vec<f32> {
        match self {
            Self::CC(a) => a.evaluate_clusters(g),
            Self::GN(a) => a.evaluate_clusters(g),
            Self::PC(a) => a.evaluate_clusters(g),
            Self::SC(a) => a.evaluate_clusters(g),
            Self::SP(a) => a.evaluate_clusters(g),
            Self::VD(a) => a.evaluate_clusters(g),
        }
    }

    /// Whether to normalize anomaly scores by cluster or by point.
    #[must_use]
    pub fn normalize_by_cluster<U: Number>(&self) -> bool {
        match self {
            Self::CC(a) => a.normalize_by_cluster(),
            Self::GN(a) => a.normalize_by_cluster(),
            Self::PC(a) => a.normalize_by_cluster(),
            Self::SC(a) => a.normalize_by_cluster(),
            Self::SP(a) => a.normalize_by_cluster(),
            Self::VD(a) => a.normalize_by_cluster(),
        }
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
    pub fn evaluate_points<U: Number>(&self, g: &mut Graph<U>) -> Vec<f32> {
        match self {
            Self::CC(a) => a.evaluate_points(g),
            Self::GN(a) => a.evaluate_points(g),
            Self::PC(a) => a.evaluate_points(g),
            Self::SC(a) => a.evaluate_points(g),
            Self::SP(a) => a.evaluate_points(g),
            Self::VD(a) => a.evaluate_points(g),
        }
    }

    /// Normalize the scores using the Error Function.
    #[must_use]
    pub fn normalize_scores(scores: &[f32]) -> Vec<f32> {
        let mean = utils::mean(scores);
        let sd = utils::standard_deviation(scores);
        utils::normalize_1d(scores, mean, sd)
    }
}

/// A trait for an algorithm in the CHAODA ensemble.
trait Algorithm: Default + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> {
    /// Get the name of the algorithm.
    fn name(&self) -> String;

    /// Evaluate the algorithm on a `Graph` and return a vector of scores for each
    /// `OddBall` in the `Graph`.
    ///
    /// The output vector must be the same length as the number of `OddBall`s in
    /// the `Graph`, and the order of the scores must correspond to the order of the
    /// `OddBall`s in the `Graph`.
    fn evaluate_clusters<U: Number>(&self, g: &mut Graph<U>) -> Vec<f32>;

    /// Whether to normalize anomaly scores by cluster or by point.
    fn normalize_by_cluster(&self) -> bool;

    /// Have points inherit scores from `OddBall`s.
    fn inherit_scores<U: Number>(&self, g: &Graph<U>, scores: &[f32]) -> Vec<f32> {
        let mut points_scores = vec![0.0; g.population()];
        for (&(c_start, c_car), &s) in g.iter_clusters().zip(scores.iter()) {
            for i in points_scores.iter_mut().skip(c_start).take(c_car) {
                *i = s;
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
    fn evaluate_points<U: Number>(&self, g: &mut Graph<U>) -> Vec<f32> {
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
