//! A `Node` in a `Graph` stores a reference to the `Vertex` it represents and
//! some additional information for the `Graph`-based anomaly detection
//! algorithms.

use core::hash::{Hash, Hasher};

use std::collections::HashSet;

use distances::Number;

use crate::{chaoda::Vertex, Cluster, Dataset};

use super::adjacency_list::AdjacencyList;

/// A `Node` in a `Graph` stores a reference to the `Vertex` it represents and
/// some additional information for the `Graph`-based anomaly detection
/// algorithms.
#[derive(Clone)]
pub struct Node<'a, I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> {
    /// The `Vertex` that the `Node` represents.
    vertex: &'a Vertex<I, U, D, S>,
    /// The cumulative size of the `Graph` neighborhood of the `Node` at each
    /// step of a breadth-first traversal.
    neighborhood_sizes: Vec<usize>,
}

impl<'a, I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Node<'a, I, U, D, S> {
    /// Create a new `Node` from a `Vertex` and an `AdjacencyList`.
    ///
    /// # Arguments
    ///
    /// * `vertex`: The `Vertex` that the `Node` represents.
    /// * `adjacency_list`: The `AdjacencyList` of the `Component` that the
    ///   `Vertex` belongs to.
    pub fn new(vertex: &'a Vertex<I, U, D, S>, adjacency_list: &AdjacencyList<I, U, D, Vertex<I, U, D, S>>) -> Self {
        let neighborhood_sizes = Self::compute_neighborhood_sizes(vertex, adjacency_list);

        Self {
            vertex,
            neighborhood_sizes,
        }
    }

    /// Get the cumulative size of the `Graph` neighborhood of the `Node` at
    /// each step of a breadth-first traversal.
    fn compute_neighborhood_sizes(
        vertex: &'a Vertex<I, U, D, S>,
        adjacency_list: &AdjacencyList<I, U, D, Vertex<I, U, D, S>>,
    ) -> Vec<usize> {
        let mut frontier_sizes = vec![];
        let mut visited = HashSet::new();
        let mut stack = vec![vertex];

        let adjacency_list = adjacency_list.inner();

        while let Some(v) = stack.pop() {
            if visited.contains(&v) {
                continue;
            }

            visited.insert(v);
            let new_neighbors = adjacency_list[&v]
                .iter()
                .filter(|(&u, _)| !visited.contains(u))
                .map(|(&u, _)| u)
                .collect::<Vec<_>>();
            frontier_sizes.push(new_neighbors.len());
            stack.extend(new_neighbors);
        }

        frontier_sizes
            .into_iter()
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect()
    }

    /// Get the eccentricity of the `Node`, i.e. the number of steps in the
    /// breadth-first traversal.
    pub fn eccentricity(&self) -> usize {
        self.neighborhood_sizes.len()
    }

    /// Get the accumulated child-parent cardinality ratio of the `Node`.
    pub const fn accumulated_cp_car_ratio(&self) -> f32 {
        self.vertex.accumulated_cp_car_ratio()
    }

    /// Get the anomaly detection properties of the `Node`.
    pub const fn anomaly_properties(&self) -> [f32; 6] {
        self.vertex.ratios()
    }

    /// Get the cumulative size of the `Graph` neighborhood of the `Node` at
    /// each step of a breadth-first traversal.
    pub const fn neighborhood_sizes(&self) -> &Vec<usize> {
        &self.neighborhood_sizes
    }
}

impl<'a, I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Eq for Node<'a, I, U, D, S> {}

impl<'a, I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> PartialEq for Node<'a, I, U, D, S> {
    fn eq(&self, other: &Self) -> bool {
        self.vertex == other.vertex
    }
}

impl<'a, I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Hash for Node<'a, I, U, D, S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex.hash(state);
    }
}
