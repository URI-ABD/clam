//! A `Component` is a single connected subgraph of a `Graph`.

use std::collections::HashMap;

use rayon::prelude::*;

use super::{adjacency_list::AdjacencyList, graph_node::GraphNode, DistanceValue, ParVertex, Vertex};

/// A `Component` is a collection of `Vertex`es that are all reachable from
/// each other in a `Graph`. It is a single connected subgraph of the `Graph`.
pub struct Component<'a, T: DistanceValue, V: Vertex<T>> {
    /// The `Vertex`es in the `Component`.
    nodes: Vec<GraphNode<'a, T, V>>,
    /// The `AdjacencyList` of the `Component`.
    adjacency_list: AdjacencyList<'a, T, V>,
    /// Diameter of the `Component`, i.e. the maximum eccentricity of its `Node`s.
    diameter: usize,
    /// The total cardinality of the `Cluster`s in the `Component`.
    population: usize,
}

impl<'a, T: DistanceValue, V: Vertex<T>> Component<'a, T, V> {
    /// Create a new `Component` from a `Vec` of `Vertex`es and the `AdjacencyList`
    /// of the `Graph`.
    ///
    /// # Arguments
    ///
    /// * `adjacency_list`: The `AdjacencyList` of the `Graph`.
    pub fn new(adjacency_list: AdjacencyList<'a, T, V>) -> Self {
        // let node_map = adjacency_list
        //     .vertices()
        //     .into_iter()
        //     .map(|v| (v, GraphNode::new(v, &adjacency_list)))
        //     .collect::<HashMap<_, _>>();\
        let nodes = adjacency_list
            .vertices()
            .into_iter()
            .map(|v| GraphNode::new(v, &adjacency_list))
            .collect::<Vec<_>>();

        let diameter = nodes.iter().map(GraphNode::eccentricity).max().unwrap_or_default();

        let population = nodes.iter().map(|n| n.as_ref().cardinality()).sum();

        Self {
            nodes,
            adjacency_list,
            diameter,
            population,
        }
    }

    /// Get the number of `Node`s in the `Component`.
    pub fn cardinality(&self) -> usize {
        self.nodes.len()
    }

    /// Get the diameter of the `Component`.
    pub const fn diameter(&self) -> usize {
        self.diameter
    }

    /// Get the total cardinality of the `Cluster`s in the `Component`.
    pub const fn population(&self) -> usize {
        self.population
    }

    /// Whether the `Component` contains any `Vertex`es.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterate over the `Vertex`es in the `Component`.
    pub fn iter_vertices(&self) -> impl Iterator<Item = &V> + '_ {
        self.nodes.iter().map(AsRef::as_ref)
    }

    /// Iterate over the edges in the `Component`.
    pub fn iter_edges(&self) -> impl Iterator<Item = (&V, &V, T)> + '_ {
        self.adjacency_list.iter_edges()
    }

    /// Iterate over the lists of neighbors of the `Node`s in the `Component`.
    pub fn iter_neighbors(&self) -> impl Iterator<Item = &HashMap<&V, T>> + '_ {
        self.adjacency_list.inner().values()
    }

    /// Iterate over the neighborhood sizes of the `Node`s in the `Component`.
    pub fn iter_neighborhood_sizes(&self) -> impl Iterator<Item = &[usize]> + '_ {
        self.nodes.iter().map(GraphNode::neighborhood_sizes)
    }

    /// Iterate over the accumulated child-parent cardinality ratio of the `Node`s
    /// in the `Component`.
    pub fn iter_accumulated_cp_car_ratios(&self) -> impl Iterator<Item = f32> + '_ {
        self.nodes.iter().map(GraphNode::accumulated_cp_car_ratio)
    }

    /// Iterate over the anomaly properties of the `Vertex`es in the `Component`.
    pub fn iter_anomaly_properties(&self) -> impl Iterator<Item = V::FeatureVector> + '_ {
        self.nodes.iter().map(GraphNode::anomaly_properties)
    }

    /// Compute the stationary probabilities of the `Node`s in the `Component`.
    ///
    /// This is the expected fraction of time that an infinite random walk on
    /// the `Component` spends at each `Node`. A high stationary probability
    /// indicates that the `Cluster` contains inliers, while a low stationary
    /// probat the transition probability from `Node` `u` to `Node` `v` to be
    /// inversely proportional to the distance from `u` to `v`, normalized
    /// across all neighbors of `u`.
    ///
    /// In the special case the `Component` containing only one `Node`, the
    /// stationary probability will be zero to mark the `Cluster` as an outlier.
    ///
    /// # Arguments
    ///
    /// * `log2_num_steps`: The logarithm base 2 of the number of steps in the
    ///   random walk.
    pub fn compute_stationary_probabilities(&self, log2_num_steps: usize) -> Vec<f32> {
        // Handle the special case of a `Component` containing only one `Node`.
        if self.cardinality() == 1 {
            return vec![0.0];
        }

        // Compute the stationary probabilities of the `Node`s in the `Component`.
        let mut transition_matrix = self.adjacency_list.transition_matrix();
        for _ in 0..log2_num_steps {
            transition_matrix = transition_matrix.dot(&transition_matrix);
        }

        // Sum each row of the transition matrix to get the stationary probabilities.
        transition_matrix.sum_axis(ndarray::Axis(1)).to_vec()
    }
}

impl<T: DistanceValue + Send + Sync, V: ParVertex<T>> Component<'_, T, V> {
    /// Iterate over the edges in the `Component`.
    pub fn par_iter_edges(&self) -> impl ParallelIterator<Item = (&V, &V, T)> + '_ {
        self.adjacency_list.par_iter_edges()
    }
}
