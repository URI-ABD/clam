//! A `Node` is a single vertex in a `Graph`. It represents one or more items from a tree, and has edges to other `Node`s in the `Graph`.

use rayon::prelude::*;

use crate::{Cluster, DistanceValue};

/// A `Node` is a single vertex in a `Graph`. It represents one or more items from a tree, and has edges to other `Node`s in the `Graph`.
#[derive(Debug, Clone)]
#[must_use]
pub struct Node<T: DistanceValue> {
    /// The indices of the items from the directly selected cluster that this node represents.
    pub(crate) direct_items: (usize, usize),
    /// The indices of the centers of ancestors of the directly selected cluster that this node represents.
    pub(crate) ancestor_centers: Vec<usize>,
    /// The edges from this node to other nodes in the graph. Each edge is represented as a tuple of the index of the other node and the distance to that node.
    pub(crate) edges: Vec<(usize, T)>,
    /// The radius of the node. This is the maximum distance from the center of the directly selected cluster to any of the items that this node represents.
    pub(crate) radius: T,
}

impl<T: DistanceValue> Node<T> {
    /// Creates a new `Node` with the given direct items.
    pub const fn from_cluster<A>(c: &Cluster<T, A>) -> Self {
        Self {
            direct_items: (c.center_index, c.center_index + c.cardinality),
            ancestor_centers: Vec::new(),
            edges: Vec::new(),
            radius: c.radius,
        }
    }

    /// Adds an edge from this node to another node with the given index and distance.
    pub fn add_edge(&mut self, other_index: usize, distance: T) {
        self.edges.push((other_index, distance));
    }

    /// Adds an ancestor center index to this node.
    pub fn add_ancestor_center(&mut self, center_index: usize) {
        self.ancestor_centers.push(center_index);
    }

    /// Returns the index of the center of the directly selected cluster that this node represents.
    pub const fn direct_center_index(&self) -> usize {
        self.direct_items.0
    }

    /// Returns the number of items that this node represents.
    pub const fn num_items(&self) -> usize {
        self.direct_items.1 - self.direct_items.0 + self.ancestor_centers.len()
    }

    /// Returns an iterator over the indices of the items represented by this node.
    pub fn iter_items(&self) -> impl Iterator<Item = usize> {
        (self.direct_items.0..self.direct_items.1).chain(self.ancestor_centers.iter().copied())
    }

    /// Returns the total number of edges from this node.
    pub const fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Returns an iterator over the edges from this node. Each edge is represented as a tuple of the index of and distance to the neighbor.
    pub fn iter_edges(&self) -> impl Iterator<Item = &(usize, T)> {
        self.edges.iter()
    }

    /// Returns the radius of this node.
    pub const fn radius(&self) -> T {
        self.radius
    }
}

impl<T: DistanceValue + Send + Sync> Node<T> {
    /// Returns a parallel iterator over the items represented by this node.
    pub fn par_iter_items(&self) -> impl ParallelIterator<Item = usize> + '_ {
        (self.direct_items.0..self.direct_items.1)
            .into_par_iter()
            .chain(self.ancestor_centers.par_iter().copied())
    }

    /// Returns a parallel iterator over the edges from this node. Each edge is represented as a tuple of the index of and distance to the neighbor.
    pub fn par_iter_edges(&self) -> impl ParallelIterator<Item = &(usize, T)> {
        self.edges.par_iter()
    }
}
