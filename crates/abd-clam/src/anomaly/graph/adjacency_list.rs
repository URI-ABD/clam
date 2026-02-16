//! An `AdjacencyList` is a map from a `Vertex` to a map from each neighbor to
//! the distance between them.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::{Dataset, DistanceValue, ParDataset};

use super::{ParVertex, Vertex};

/// An `AdjacencyList` is a map from a `Vertex` to a map from each neighbor to
/// the distance between them.
pub struct AdjacencyList<'a, T: DistanceValue, V: Vertex<T>>(HashMap<&'a V, HashMap<&'a V, T>>);

impl<'a, T: DistanceValue, V: Vertex<T>> AdjacencyList<'a, T, V> {
    /// Create new `AdjacencyList`s for each `Component` in a `Graph`.
    ///
    /// # Arguments
    ///
    /// * data: The `Dataset` that the `Vertex`s are based on.
    /// * vertices: The `Vertex`es to create the `AdjacencyList` from.
    pub fn new<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(vertices: &[&'a V], data: &D, metric: &M) -> Vec<Self> {
        // TODO: This is a naive implementation of creating an adjacency list.
        // We can improve this by using the search functionality from CAKES.
        let inner = vertices
            .iter()
            .enumerate()
            .map(|(i, &u)| {
                let neighbors = vertices
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| i != j)
                    .filter_map(|(_, &v)| {
                        let (ru, rv) = (u.radius(), v.radius());
                        let d = data.one_to_one(u.arg_center(), v.arg_center(), metric);
                        if d <= ru + rv {
                            Some((v, d))
                        } else {
                            None
                        }
                    })
                    .collect();
                (u, neighbors)
            })
            .collect();

        let adjacency_list = Self(inner);
        let [mut c, mut other] = adjacency_list.partition();
        let mut components = vec![c];
        while !other.0.is_empty() {
            [c, other] = other.partition();
            components.push(c);
        }

        components
    }

    /// Partition the `AdjacencyList` into two `AdjacencyList`s.
    ///
    /// The first `AdjacencyList` contains a `Component`s worth of `Vertex`es
    /// and their neighbors. The second `AdjacencyList` contains the remaining
    /// `Vertex`es and their neighbors.
    ///
    /// This method can be used repeatedly to partition a full `AdjacencyList`
    /// into individual connected-`Component`s.
    fn partition(mut self) -> [Self; 2] {
        let mut visited = HashSet::new();

        let start = self
            .0
            .keys()
            .next()
            .copied()
            .unwrap_or_else(|| unreachable!("We never call `partition` on an empty `AdjacencyList`"));
        let mut stack = vec![start];

        while let Some(u) = stack.pop() {
            // Check if the `Vertex` has already been visited.
            if visited.contains(&u) {
                continue;
            }

            // Mark the `Vertex` as visited.
            visited.insert(u);

            // Add the neighbors of the `Vertex` to the stack.
            stack.extend(
                self.0[&u]
                    .iter()
                    .filter(|(&v, _)| !visited.contains(v))
                    .map(|(&v, _)| v),
            );
        }

        let (inner, other): (HashMap<_, _>, HashMap<_, _>) = self.0.into_iter().partition(|(u, _)| visited.contains(u));

        self.0 = inner;
        let other = Self(other);

        [self, other]
    }

    /// Get the inner `HashMap`.
    pub const fn inner(&self) -> &HashMap<&V, HashMap<&V, T>> {
        &self.0
    }

    /// Compute the transition probability matrix of the `AdjacencyList`.
    pub fn transition_matrix(&self) -> ndarray::Array2<f32> {
        let n = self.0.len();

        // Compute the (flattened) transition matrix.
        let mut transition_matrix = vec![0_f32; n * n];
        for (i, (_, neighbors)) in self.0.iter().enumerate() {
            for (j, (_, d)) in neighbors.iter().enumerate() {
                transition_matrix[i * n + j] = d
                    .to_f32()
                    .unwrap_or_else(|| unreachable!("Could not convert distance to f32"))
                    .recip();
            }
        }

        // Convert the transition matrix to a 2d array.
        let mut transition_matrix = ndarray::Array2::from_shape_vec((n, n), transition_matrix)
            .unwrap_or_else(|e| unreachable!("We created the array with the correct shape: {e}"));

        // Normalize the transition matrix.
        for i in 0..n {
            let row_sum = transition_matrix.row(i).sum();
            transition_matrix.row_mut(i).mapv_inplace(|x| x / row_sum);
        }

        transition_matrix
    }

    /// Iterate over the `Vertex`s in the `AdjacencyList`.
    pub fn vertices(&self) -> Vec<&'a V> {
        self.0.keys().copied().collect()
    }

    /// Iterate over the edges in the `AdjacencyList`.
    pub fn iter_edges(&self) -> impl Iterator<Item = (&V, &V, T)> + '_ {
        self.0
            .iter()
            .flat_map(|(&u, neighbors)| neighbors.iter().map(move |(&v, d)| (u, v, *d)))
    }
}

impl<'a, T: DistanceValue + Send + Sync, V: ParVertex<T>> AdjacencyList<'a, T, V> {
    /// Parallel version of [`AdjacencyList::new`](crate::chaoda::graph::adjacency_list::AdjacencyList::new).
    pub fn par_new<I: Send + Sync, D: ParDataset<I>, M: (Fn(&I, &I) -> T) + Send + Sync>(
        vertices: &[&'a V],
        data: &D,
        metric: &M,
    ) -> Vec<Self> {
        // TODO: This is a naive implementation of creating an adjacency list.
        // We can improve this by using the search functionality from CAKES.
        let inner = vertices
            .par_iter()
            .enumerate()
            .map(|(i, &u)| {
                let neighbors = vertices
                    .par_iter()
                    .enumerate()
                    .filter(|&(j, _)| i != j)
                    .filter_map(|(_, &v)| {
                        let (ru, rv) = (u.radius(), v.radius());
                        let d = data.one_to_one(u.arg_center(), v.arg_center(), metric);
                        if d <= ru + rv {
                            Some((v, d))
                        } else {
                            None
                        }
                    })
                    .collect();
                (u, neighbors)
            })
            .collect();

        let adjacency_list = Self(inner);
        let [mut c, mut other] = adjacency_list.partition();
        let mut components = vec![c];
        while !other.0.is_empty() {
            [c, other] = other.partition();
            components.push(c);
        }

        components
    }

    /// Iterate over the edges in the `AdjacencyList`.
    pub fn par_iter_edges(&self) -> impl ParallelIterator<Item = (&V, &V, T)> + '_ {
        self.0
            .par_iter()
            .flat_map(|(&u, neighbors)| neighbors.par_iter().map(move |(&v, d)| (u, v, *d)))
    }
}
