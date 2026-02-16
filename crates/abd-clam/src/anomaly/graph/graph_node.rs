//! A `Node` in a `Graph` stores a reference to the `Vertex` it represents and
//! some additional information for the `Graph`-based anomaly detection
//! algorithms.

use std::collections::HashSet;

use crate::DistanceValue;

use super::{adjacency_list::AdjacencyList, Vertex};

/// A `Node` in a `Graph` stores a reference to the `Vertex` it represents and
/// some additional information for the `Graph`-based anomaly detection
/// algorithms.
#[derive(Clone)]
pub struct GraphNode<'a, T: DistanceValue, V: Vertex<T>> {
    /// The `Vertex` that the `Node` represents.
    vertex: &'a V,
    /// The cumulative size of the `Graph` neighborhood of the `Node` at each
    /// step of a breadth-first traversal.
    neighborhood_sizes: Vec<usize>,
    /// Ghost marker to retain the cluster type.
    _marker: core::marker::PhantomData<T>,
}

impl<'a, T: DistanceValue, V: Vertex<T>> GraphNode<'a, T, V> {
    /// Create a new `Node` from a `Vertex` and an `AdjacencyList`.
    ///
    /// # Arguments
    ///
    /// * `vertex`: The `Vertex` that the `Node` represents.
    /// * `adjacency_list`: The `AdjacencyList` of the `Component` that the
    ///   `Vertex` belongs to.
    pub fn new(vertex: &'a V, adjacency_list: &AdjacencyList<T, V>) -> Self {
        let neighborhood_sizes = Self::compute_neighborhood_sizes(vertex, adjacency_list);

        Self {
            vertex,
            neighborhood_sizes,
            _marker: core::marker::PhantomData,
        }
    }

    /// Get the cumulative size of the `Graph` neighborhood of the `Node` at
    /// each step of a breadth-first traversal.
    fn compute_neighborhood_sizes(vertex: &'a V, adjacency_list: &AdjacencyList<T, V>) -> Vec<usize> {
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
    pub fn accumulated_cp_car_ratio(&self) -> f32 {
        self.vertex.accumulated_cp_cardinality_ratio()
    }

    /// Get the anomaly detection properties of the `Node`.
    pub fn anomaly_properties(&self) -> V::FeatureVector {
        self.vertex.feature_vector()
    }

    /// Get the cumulative size of the `Graph` neighborhood of the `Node` at
    /// each step of a breadth-first traversal.
    pub fn neighborhood_sizes(&self) -> &[usize] {
        &self.neighborhood_sizes
    }
}

impl<T: DistanceValue, V: Vertex<T>> Eq for GraphNode<'_, T, V> {}

impl<T: DistanceValue, V: Vertex<T>> PartialEq for GraphNode<'_, T, V> {
    fn eq(&self, other: &Self) -> bool {
        self.vertex == other.vertex
    }
}

impl<T: DistanceValue, V: Vertex<T>> core::hash::Hash for GraphNode<'_, T, V> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.vertex.hash(state);
    }
}

impl<T: DistanceValue, V: Vertex<T>> AsRef<V> for GraphNode<'_, T, V> {
    fn as_ref(&self) -> &V {
        self.vertex
    }
}
