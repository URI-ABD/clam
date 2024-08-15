//! A collection of `Vertex`es and `Edge`s from a CLAM-tree. The `Graph` is used
//! for anomaly detection, dimension reduction, and visualization.

use core::cmp::Reverse;

use std::collections::BinaryHeap;

use distances::Number;
use ordered_float::OrderedFloat;

use crate::{Cluster, Dataset};

use super::Vertex;

mod component;

pub use component::{Component, Neighbors};

/// A `Graph` is a collection of `Vertex`es.
pub struct Graph<'a, I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> {
    /// The collection of `Component`s in the `Graph`.
    components: Vec<Component<'a, I, U, D, S>>,
    /// Cumulative populations of the `Component`s in the `Graph`.
    populations: Vec<usize>,
}

// , C: OddBall<U>, const N: usize
impl<'a, I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Graph<'a, I, U, D, S> {
    /// Create a new `Graph` from a `Tree`.
    ///
    /// # Arguments
    ///
    /// * `tree`: The `Tree` to create the `Graph` from.
    /// * `cluster_scorer`: A function that scores `OddBall`s.
    /// * `min_depth`: The minimum depth at which to consider a `OddBall`.
    pub fn from_tree(
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
        // We use `Reverse` on `OddBall` so that we can bias towards selecting shallower `OddBall`s.
        // `OddBall`s are selected by highest score and then by shallowest depth.
        let mut candidates = clusters
            .into_iter()
            .zip(scores.into_iter().map(OrderedFloat))
            .filter(|(c, _)| c.is_leaf() || c.depth() >= min_depth)
            .map(|(c, s)| (s, Reverse(c)))
            .collect::<BinaryHeap<_>>();

        let mut clusters = vec![];
        while let Some((_, Reverse(v))) = candidates.pop() {
            clusters.push(v);
            // Remove `OddBall`s that are ancestors or descendants of the selected `OddBall`, so as not to have duplicates
            // in the `Graph`.
            candidates.retain(|&(_, Reverse(other))| !(v.is_descendant_of(other) || other.is_descendant_of(v)));
        }

        Self::from_vertices(&clusters, data)
    }

    /// Create a new `Graph` from a collection of `OddBall`s.
    pub fn from_vertices(vertices: &[&'a Vertex<I, U, D, S>], data: &D) -> Self {
        let components = Component::new(vertices, data);
        let populations = components
            .iter()
            .map(Component::population)
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        Self {
            components,
            populations,
        }
    }

    /// Iterate over the `OddBall`s in the `Graph`.
    pub fn iter_clusters(&self) -> impl Iterator<Item = &Vertex<I, U, D, S>> {
        self.components.iter().flat_map(Component::iter_clusters)
    }

    /// Iterate over the lists of neighbors of the `OddBall`s in the `Graph`.
    pub fn iter_neighbors(&self) -> impl Iterator<Item = &Neighbors<&Vertex<I, U, D, S>, U>> {
        self.components.iter().flat_map(Component::iter_neighbors)
    }

    /// Iterate over the anomaly properties of the `OddBall`s in the `Graph`.
    pub fn iter_anomaly_properties(&self) -> impl Iterator<Item = &[f32; 6]> {
        self.components.iter().flat_map(Component::iter_anomaly_properties)
    }

    /// Get the diameter of the `Graph`.
    pub fn diameter(&mut self) -> usize {
        self.components.iter_mut().map(Component::diameter).max().unwrap_or(0)
    }

    /// Get the neighborhood sizes of all `OddBall`s in the `Graph`.
    pub fn neighborhood_sizes(&mut self) -> impl Iterator<Item = &Vec<usize>> + '_ {
        self.components
            .iter_mut()
            .map(Component::neighborhood_sizes)
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
    }

    /// Get the total number of points in the `Graph`.
    #[must_use]
    pub fn population(&self) -> usize {
        self.populations.last().copied().unwrap_or(0)
    }

    /// Iterate over the `Component`s in the `Graph`.
    pub fn iter_components(&self) -> impl Iterator<Item = &Component<I, U, D, S>> {
        self.components.iter()
    }

    /// Compute the stationary probability of each `OddBall` in the `Graph`.
    #[must_use]
    pub fn compute_stationary_probabilities(&self, num_steps: usize) -> Vec<f32> {
        self.components
            .iter()
            .flat_map(|c| c.compute_stationary_probabilities(num_steps))
            .collect()
    }

    /// Get the accumulated child-parent cardinality ratio of each `OddBall` in the `Graph`.
    #[must_use]
    pub fn accumulated_cp_car_ratios(&self) -> Vec<f32> {
        self.components
            .iter()
            .flat_map(Component::accumulated_cp_car_ratios)
            .collect()
    }
}
