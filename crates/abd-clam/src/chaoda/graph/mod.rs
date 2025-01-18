//! A collection of `Vertex`es and `Edge`s from a CLAM-tree. The `Graph` is used
//! for anomaly detection, dimension reduction, and visualization.

use core::cmp::Reverse;

use std::collections::HashMap;

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, dataset::ParDataset, metric::ParMetric, Cluster, Dataset, Metric, SizedHeap};

use super::Vertex;

mod adjacency_list;
mod component;
mod node;

pub use component::Component;

/// A `Graph` is a collection of `Vertex`es.
pub struct Graph<'a, T: Number, S: Cluster<T>> {
    /// The collection of `Component`s in the `Graph`.
    components: Vec<Component<'a, T, S>>,
    /// The total number of points in the `Graph`.
    population: usize,
    /// The number of vertices in the `Graph`.
    cardinality: usize,
    /// The diameter of the `Graph`.
    diameter: usize,
}

impl<'a, T: Number, S: Cluster<T>> Graph<'a, T, S> {
    /// Create a new `Graph` from a root `Vertex` in a tree using a uniform
    /// depth from the tree.
    ///
    /// # Arguments
    ///
    /// * `root`: The root `Vertex` of the tree from which to create the `Graph`.
    /// * `depth`: The uniform depth at which to consider a `Vertex`.
    /// * `min_depth`: The minimum depth at which to consider a `Vertex`.
    pub fn from_root_uniform_depth<I, D: Dataset<I>, M: Metric<I, T>>(
        root: &'a Vertex<T, S>,
        data: &D,
        metric: &M,
        depth: usize,
        min_depth: usize,
    ) -> Self
    where
        T: 'a,
    {
        let cluster_scorer = |clusters: &[&'a Vertex<T, S>]| {
            clusters
                .iter()
                .map(|c| {
                    if c.depth() == depth || (c.is_leaf() && c.depth() < depth) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
        };
        Self::from_root(root, data, metric, cluster_scorer, min_depth)
    }

    /// Create a new `Graph` from a root `Vertex` in a tree.
    ///
    /// # Arguments
    ///
    /// * `root`: The root `Vertex` of the tree from which to create the `Graph`.
    /// * `cluster_scorer`: A function that scores `Vertex`es.
    /// * `min_depth`: The minimum depth at which to consider a `Vertex`.
    pub fn from_root<I, D: Dataset<I>, M: Metric<I, T>>(
        root: &'a Vertex<T, S>,
        data: &D,
        metric: &M,
        cluster_scorer: impl Fn(&[&'a Vertex<T, S>]) -> Vec<f32>,
        min_depth: usize,
    ) -> Self
    where
        T: 'a,
    {
        let clusters = root.subtree();
        let scores = cluster_scorer(&clusters);

        // We use `OrderedFloat` to have the `Ord` trait implemented for `f64` so that we can use it in a `BinaryHeap`.
        // We use `Reverse` on `Vertex` so that we can bias towards selecting shallower `Vertex`es.
        // `Vertex`es are selected by highest score and then by shallowest depth.
        let mut candidates = clusters
            .into_iter()
            .zip(scores)
            .filter(|(c, _)| c.is_leaf() || c.depth() >= min_depth)
            .map(|(c, s)| (s, Reverse(c)))
            .collect::<SizedHeap<_>>();

        let mut clusters = vec![];
        while let Some((_, Reverse(v))) = candidates.pop() {
            clusters.push(v);
            // Remove `Vertex`es that are ancestors or descendants of the selected `Vertex`, so as not to have duplicates
            // in the `Graph`.
            candidates.retain(|&(_, Reverse(other))| !(v.is_descendant_of(other) || other.is_descendant_of(v)));
        }

        Self::from_vertices(&clusters, data, metric)
    }

    /// Create a new `Graph` from a collection of `Vertex`es.
    pub fn from_vertices<I, D: Dataset<I>, M: Metric<I, T>>(
        vertices: &[&'a Vertex<T, S>],
        data: &D,
        metric: &M,
    ) -> Self {
        let components = adjacency_list::AdjacencyList::new(vertices, data, metric)
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
    pub fn iter_clusters(&self) -> impl Iterator<Item = &Vertex<T, S>> {
        self.components.iter().flat_map(Component::iter_vertices)
    }

    /// Iterate over the edges in the `Graph`.
    pub fn iter_edges(&self) -> impl Iterator<Item = (&Vertex<T, S>, &Vertex<T, S>, T)> + '_ {
        self.components.iter().flat_map(Component::iter_edges)
    }

    /// Iterate over the lists of neighbors of the `Vertex`es in the `Graph`.
    pub fn iter_neighbors(&self) -> impl Iterator<Item = &HashMap<&Vertex<T, S>, T>> + '_ {
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
    pub fn iter_components(&self) -> impl Iterator<Item = &Component<T, S>> {
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

impl<'a, T: Number, S: ParCluster<T>> Graph<'a, T, S> {
    /// Parallel version of [`Graph::from_root_uniform_depth`](crate::chaoda::graph::Graph::from_root_uniform_depth).
    pub fn par_from_root_uniform_depth<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>>(
        root: &'a Vertex<T, S>,
        data: &D,
        metric: &M,
        depth: usize,
        min_depth: usize,
    ) -> Self
    where
        T: 'a,
    {
        let cluster_scorer = |clusters: &[&'a Vertex<T, S>]| {
            clusters
                .iter()
                .map(|c| {
                    if c.depth() == depth || (c.is_leaf() && c.depth() < depth) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
        };
        Self::par_from_root(root, data, metric, cluster_scorer, min_depth)
    }
    /// Parallel version of [`Graph::from_root`](crate::chaoda::graph::Graph::from_root).
    pub fn par_from_root<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>>(
        root: &'a Vertex<T, S>,
        data: &D,
        metric: &M,
        cluster_scorer: impl Fn(&[&'a Vertex<T, S>]) -> Vec<f32>,
        min_depth: usize,
    ) -> Self
    where
        T: 'a,
    {
        let clusters = root.subtree();
        let scores = cluster_scorer(&clusters);

        // We use `OrderedFloat` to have the `Ord` trait implemented for `f64` so that we can use it in a `BinaryHeap`.
        // We use `Reverse` on `Vertex` so that we can bias towards selecting shallower `Vertex`es.
        // `Vertex`es are selected by highest score and then by shallowest depth.
        let mut candidates = clusters
            .into_iter()
            .zip(scores)
            .filter(|(c, _)| c.is_leaf() || c.depth() >= min_depth)
            .map(|(c, s)| (s, Reverse(c)))
            .collect::<SizedHeap<_>>();

        let mut clusters = vec![];
        while let Some((_, Reverse(v))) = candidates.pop() {
            clusters.push(v);
            // Remove `Vertex`es that are ancestors or descendants of the selected `Vertex`, so as not to have duplicates
            // in the `Graph`.
            candidates.retain(|&(_, Reverse(other))| !(v.is_descendant_of(other) || other.is_descendant_of(v)));
        }

        Self::par_from_vertices(&clusters, data, metric)
    }

    /// Parallel version of [`Graph::from_vertices`](crate::chaoda::graph::Graph::from_vertices).
    pub fn par_from_vertices<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>>(
        vertices: &[&'a Vertex<T, S>],
        data: &D,
        metric: &M,
    ) -> Self {
        let components = adjacency_list::AdjacencyList::par_new(vertices, data, metric)
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

    /// Parallel version of [`Graph::iter_clusters`](crate::chaoda::graph::Graph::iter_clusters).
    #[must_use]
    pub fn par_compute_stationary_probabilities(&self, log2_num_steps: usize) -> Vec<f32> {
        self.components
            .par_iter()
            .flat_map(|c| c.compute_stationary_probabilities(log2_num_steps))
            .collect()
    }
}
