//! A `Component` is a single connected subgraph of a `Graph`.

use std::collections::HashMap;

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, Cluster};

use super::{adjacency_list::AdjacencyList, node::Node, Vertex};

/// A `Component` is a collection of `Node`s that are connected by edges in a
/// `Graph`. Every `Node` in a `Component` is reachable from every other `Node`
/// via a path along edges.
pub struct Component<'a, T: Number, S: Cluster<T>> {
    /// A map from each `Vertex` to the `Node` that represents it.
    #[allow(clippy::type_complexity)]
    node_map: HashMap<&'a Vertex<T, S>, Node<'a, T, S>>,
    /// The `AdjacencyList` of the `Component`.
    adjacency_list: AdjacencyList<'a, T, Vertex<T, S>>,
    // adjacency_list: HashMap<Node<'a, I, U, D, S>, HashMap<&'a Node<'a, I, U, D, S>, U>>,
    /// Diameter of the `Component`, i.e. the maximum eccentricity of its `Node`s.
    diameter: usize,
    /// The total cardinality of the `Cluster`s in the `Component`.
    population: usize,
}

impl<'a, T: Number, S: Cluster<T>> Component<'a, T, S> {
    /// Create a new `Component` from a `Vec` of `Vertex`es and the `AdjacencyList`
    /// of the `Graph`.
    ///
    /// # Arguments
    ///
    /// * `adjacency_list`: The `AdjacencyList` of the `Graph`.
    pub fn new(adjacency_list: AdjacencyList<'a, T, Vertex<T, S>>) -> Self {
        let node_map = adjacency_list
            .clusters()
            .into_iter()
            .map(|v| (v, Node::new(v, &adjacency_list)))
            .collect::<HashMap<_, _>>();

        let diameter = node_map.values().map(Node::eccentricity).max().unwrap_or_default();

        let population = node_map.keys().map(|v| v.cardinality()).sum();

        Self {
            node_map,
            adjacency_list,
            diameter,
            population,
        }
    }

    /// Get the number of `Node`s in the `Component`.
    pub fn cardinality(&self) -> usize {
        self.node_map.len()
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
        self.node_map.is_empty()
    }

    /// Iterate over the `Vertex`es in the `Component`.
    pub fn iter_vertices(&self) -> impl Iterator<Item = &Vertex<T, S>> + '_ {
        self.node_map.keys().copied()
    }

    /// Iterate over the edges in the `Component`.
    pub fn iter_edges(&self) -> impl Iterator<Item = (&Vertex<T, S>, &Vertex<T, S>, T)> + '_ {
        self.adjacency_list.iter_edges()
    }

    /// Iterate over the lists of neighbors of the `Node`s in the `Component`.
    pub fn iter_neighbors(&self) -> impl Iterator<Item = &HashMap<&Vertex<T, S>, T>> + '_ {
        self.adjacency_list.inner().values()
    }

    /// Iterate over the neighborhood sizes of the `Node`s in the `Component`.
    pub fn iter_neighborhood_sizes(&self) -> impl Iterator<Item = &Vec<usize>> + '_ {
        self.node_map.values().map(Node::neighborhood_sizes)
    }

    /// Iterate over the accumulated child-parent cardinality ratio of the `Node`s
    /// in the `Component`.
    pub fn iter_accumulated_cp_car_ratios(&self) -> impl Iterator<Item = f32> + '_ {
        self.node_map.values().map(Node::accumulated_cp_car_ratio)
    }

    /// Iterate over the anomaly properties of the `Vertex`es in the `Component`.
    pub fn iter_anomaly_properties(&self) -> impl Iterator<Item = [f32; 6]> + '_ {
        self.node_map.values().map(Node::anomaly_properties)
    }

    /// Compute the stationary probabilities of the `Node`s in the `Component`.
    ///
    /// This is the expected fraction of time that an infinite random walk on
    /// the `Component` spends at each `Node`. A high stationary probability
    /// indicates that the `Cluster` contains inliers, while a low stationary
    /// probability indicates that the `Cluster` contains outliers.
    ///
    /// We set the transition probability from `Node` `u` to `Node` `v` to be
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

impl<'a, T: Number, S: ParCluster<T>> Component<'a, T, S> {
    /// Parallel version of [`Component::new`](crate::chaoda::graph::component::Component::new).
    pub fn par_new(adjacency_list: AdjacencyList<'a, T, Vertex<T, S>>) -> Self {
        let node_map = adjacency_list
            .clusters()
            .into_par_iter()
            .map(|v| (v, Node::new(v, &adjacency_list)))
            .collect::<HashMap<_, _>>();

        let diameter = node_map.values().map(Node::eccentricity).max().unwrap_or_default();

        let population = node_map.keys().map(|v| v.cardinality()).sum();

        Self {
            node_map,
            adjacency_list,
            diameter,
            population,
        }
    }
}
