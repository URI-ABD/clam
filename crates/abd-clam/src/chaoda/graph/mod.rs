//! A collection of `Vertex`es and `Edge`s from a CLAM-tree. The `Graph` is used
//! for anomaly detection, dimension reduction, and visualization.

use core::cmp::Reverse;

use std::collections::{BinaryHeap, HashMap};

use distances::Number;
use ordered_float::OrderedFloat;
use rayon::prelude::*;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

use super::Vertex;

mod adjacency_list;
mod component;
mod node;

pub use component::Component;

/// A `Graph` is a collection of `Vertex`es.
pub struct Graph<'a, I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> {
    /// The collection of `Component`s in the `Graph`.
    components: Vec<Component<'a, I, U, D, S>>,
    /// The total number of points in the `Graph`.
    population: usize,
    /// The number of vertices in the `Graph`.
    cardinality: usize,
    /// The diameter of the `Graph`.
    diameter: usize,
}

impl<'a, I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Graph<'a, I, U, D, S> {
    /// Create a new `Graph` from a root `Vertex` in a tree.
    ///
    /// # Arguments
    ///
    /// * `root`: The root `Vertex` of the tree from which to create the `Graph`.
    /// * `cluster_scorer`: A function that scores `Vertex`es.
    /// * `min_depth`: The minimum depth at which to consider a `Vertex`.
    pub fn from_root(
        root: &'a Vertex<I, U, D, S>,
        data: &D,
        cluster_scorer: impl Fn(&[&'a Vertex<I, U, D, S>]) -> Vec<f32>,
        min_depth: usize,
    ) -> Self
    where
        U: 'a,
    {
        let clusters = root.subtree();
        let scores = cluster_scorer(&clusters);

        // We use `OrderedFloat` to have the `Ord` trait implemented for `f64` so that we can use it in a `BinaryHeap`.
        // We use `Reverse` on `Vertex` so that we can bias towards selecting shallower `Vertex`es.
        // `Vertex`es are selected by highest score and then by shallowest depth.
        let mut candidates = clusters
            .into_iter()
            .zip(scores.into_iter().map(OrderedFloat))
            .filter(|(c, _)| c.is_leaf() || c.depth() >= min_depth)
            .map(|(c, s)| (s, Reverse(c)))
            .collect::<BinaryHeap<_>>();

        let mut clusters = vec![];
        while let Some((_, Reverse(v))) = candidates.pop() {
            clusters.push(v);
            // Remove `Vertex`es that are ancestors or descendants of the selected `Vertex`, so as not to have duplicates
            // in the `Graph`.
            candidates.retain(|&(_, Reverse(other))| !(v.is_descendant_of(other) || other.is_descendant_of(v)));
        }

        Self::from_vertices(&clusters, data)
    }

    /// Create a new `Graph` from a collection of `Vertex`es.
    pub fn from_vertices(vertices: &[&'a Vertex<I, U, D, S>], data: &D) -> Self {
        let components = adjacency_list::AdjacencyList::new(vertices, data)
            .into_iter()
            .map(Component::new)
            .collect::<Vec<_>>();

        let population = vertices.iter().map(|v| v.cardinality()).sum();
        let cardinality = components.iter().map(Component::cardinality).sum();
        let diameter = components.iter().map(Component::diameter).max().unwrap_or_default();
        Self {
            components,
            population,
            cardinality,
            diameter,
        }
    }

    /// Cet teh number of `Vertex`es in the `Graph`.
    #[must_use]
    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Get the total number of points in the `Graph`.
    #[must_use]
    pub const fn population(&self) -> usize {
        self.population
    }

    /// Get the diameter of the `Graph`.
    #[must_use]
    pub const fn diameter(&self) -> usize {
        self.diameter
    }

    /// Iterate over the `Vertex`es in the `Graph`.
    pub fn iter_clusters(&self) -> impl Iterator<Item = &Vertex<I, U, D, S>> {
        self.components.iter().flat_map(Component::iter_vertices)
    }

    /// Iterate over the edges in the `Graph`.
    pub fn iter_edges(&self) -> impl Iterator<Item = (&Vertex<I, U, D, S>, &Vertex<I, U, D, S>, U)> + '_ {
        self.components.iter().flat_map(Component::iter_edges)
    }

    /// Iterate over the lists of neighbors of the `Vertex`es in the `Graph`.
    pub fn iter_neighbors(&self) -> impl Iterator<Item = &HashMap<&Vertex<I, U, D, S>, U>> + '_ {
        self.components.iter().flat_map(Component::iter_neighbors)
    }

    /// Iterate over the anomaly properties of the `Vertex`es in the `Graph`.
    pub fn iter_anomaly_properties(&self) -> impl Iterator<Item = [f32; 6]> + '_ {
        self.components.iter().flat_map(Component::iter_anomaly_properties)
    }

    /// Get the neighborhood sizes of all `Vertex`es in the `Graph`.
    pub fn iter_neighborhood_sizes(&self) -> impl Iterator<Item = &Vec<usize>> + '_ {
        self.components.iter().flat_map(Component::iter_neighborhood_sizes)
    }

    /// Iterate over the `Component`s in the `Graph`.
    pub fn iter_components(&self) -> impl Iterator<Item = &Component<I, U, D, S>> {
        self.components.iter()
    }

    /// Get the accumulated child-parent cardinality ratio of each `Vertex` in the `Graph`.
    pub fn iter_accumulated_cp_car_ratios(&self) -> impl Iterator<Item = f32> + '_ {
        self.components
            .iter()
            .flat_map(Component::iter_accumulated_cp_car_ratios)
    }

    /// Compute the stationary probability of each `Vertex` in the `Graph`.
    #[must_use]
    pub fn compute_stationary_probabilities(&self, log2_num_steps: usize) -> Vec<f32> {
        self.components
            .iter()
            .flat_map(|c| c.compute_stationary_probabilities(log2_num_steps))
            .collect()
    }
}

impl<'a, I: Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<I, U, D>> Graph<'a, I, U, D, S> {
    /// Parallel version of `Graph::from_root`.
    pub fn par_from_root(
        root: &'a Vertex<I, U, D, S>,
        data: &D,
        cluster_scorer: impl Fn(&[&'a Vertex<I, U, D, S>]) -> Vec<f32>,
        min_depth: usize,
    ) -> Self
    where
        U: 'a,
    {
        let clusters = root.subtree();
        let scores = cluster_scorer(&clusters);

        // We use `OrderedFloat` to have the `Ord` trait implemented for `f64` so that we can use it in a `BinaryHeap`.
        // We use `Reverse` on `Vertex` so that we can bias towards selecting shallower `Vertex`es.
        // `Vertex`es are selected by highest score and then by shallowest depth.
        let mut candidates = clusters
            .into_iter()
            .zip(scores.into_iter().map(OrderedFloat))
            .filter(|(c, _)| c.is_leaf() || c.depth() >= min_depth)
            .map(|(c, s)| (s, Reverse(c)))
            .collect::<BinaryHeap<_>>();

        let mut clusters = vec![];
        while let Some((_, Reverse(v))) = candidates.pop() {
            clusters.push(v);
            // Remove `Vertex`es that are ancestors or descendants of the selected `Vertex`, so as not to have duplicates
            // in the `Graph`.
            candidates.retain(|&(_, Reverse(other))| !(v.is_descendant_of(other) || other.is_descendant_of(v)));
        }

        Self::par_from_vertices(&clusters, data)
    }

    /// Create a new `Graph` from a collection of `Vertex`es.
    pub fn par_from_vertices(vertices: &[&'a Vertex<I, U, D, S>], data: &D) -> Self {
        let components = adjacency_list::AdjacencyList::par_new(vertices, data)
            .into_par_iter()
            .map(Component::new)
            .collect::<Vec<_>>();

        let population = vertices.iter().map(|v| v.cardinality()).sum();
        let cardinality = components.iter().map(Component::cardinality).sum();
        let diameter = components.iter().map(Component::diameter).max().unwrap_or_default();
        Self {
            components,
            population,
            cardinality,
            diameter,
        }
    }

    /// Compute the stationary probability of each `Vertex` in the `Graph`.
    #[must_use]
    pub fn par_compute_stationary_probabilities(&self, log2_num_steps: usize) -> Vec<f32> {
        self.components
            .par_iter()
            .flat_map(|c| c.compute_stationary_probabilities(log2_num_steps))
            .collect()
    }
}
