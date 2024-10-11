//! An `AdjacencyList` is a map from a `Cluster` to a map from each neighbor to
//! the distance between them.

use std::collections::{HashMap, HashSet};

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

/// An `AdjacencyList` is a map from a `Cluster` to a map from each neighbor to
/// the distance between them.
pub struct AdjacencyList<'a, I, U: Number, D: Dataset<I, U>, C: Cluster<I, U, D>> {
    /// The inner `HashMap` of the `AdjacencyList`.
    inner: HashMap<&'a C, HashMap<&'a C, U>>,
    /// Phantom data to keep track of the types.
    _phantom: std::marker::PhantomData<(I, D)>,
}

impl<'a, I, U: Number, D: Dataset<I, U>, C: Cluster<I, U, D>> AdjacencyList<'a, I, U, D, C> {
    /// Create new `AdjacencyList`s for each `Component` in a `Graph`.
    ///
    /// # Arguments
    ///
    /// * data: The `Dataset` that the `Cluster`s are based on.
    /// * clusters: The `Cluster`s to create the `AdjacencyList` from.
    pub fn new(clusters: &[&'a C], data: &D) -> Vec<Self> {
        // TODO: This is a naive implementation of creating an adjacency list.
        // We can improve this by using the search functionality from CAKES.
        let inner = clusters
            .iter()
            .enumerate()
            .map(|(i, &u)| {
                let neighbors = clusters
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| i != j)
                    .filter_map(|(_, &v)| {
                        let (ru, rv) = (u.radius(), v.radius());
                        let d = u.distance_to_other(data, v);
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

        let c = Self {
            inner,
            _phantom: std::marker::PhantomData,
        };

        let [mut c, mut other] = c.partition();
        let mut components = vec![c];
        while !other.inner.is_empty() {
            [c, other] = other.partition();
            components.push(c);
        }

        components
    }

    /// Partition the `AdjacencyList` into two `AdjacencyList`s.
    ///
    /// The first `AdjacencyList` contains a `Component`s worth of `Cluster`s
    /// and their neighbors. The second `AdjacencyList` contains the remaining
    /// `Cluster`s and their neighbors.
    ///
    /// This method can be used repeatedly to partition a full `AdjacencyList`
    /// into individual connected-`Component`s.
    fn partition(mut self) -> [Self; 2] {
        let mut visited = HashSet::new();

        let start = self
            .inner
            .keys()
            .next()
            .copied()
            .unwrap_or_else(|| unreachable!("We never call `partition` on an empty `AdjacencyList`"));
        let mut stack = vec![start];

        while let Some(u) = stack.pop() {
            // Check if the `Cluster` has already been visited.
            if visited.contains(&u) {
                continue;
            }

            // Mark the `Cluster` as visited.
            visited.insert(u);

            // Add the neighbors of the `Cluster` to the stack.
            stack.extend(
                self.inner[&u]
                    .iter()
                    .filter(|(&v, _)| !visited.contains(v))
                    .map(|(&v, _)| v),
            );
        }

        let (inner, other): (HashMap<_, _>, HashMap<_, _>) =
            self.inner.into_iter().partition(|(u, _)| visited.contains(u));

        self.inner = inner;
        let other = Self {
            inner: other,
            _phantom: std::marker::PhantomData,
        };

        [self, other]
    }

    /// Get the inner `HashMap`.
    pub const fn inner(&self) -> &HashMap<&C, HashMap<&C, U>> {
        &self.inner
    }

    /// Compute the transition probability matrix of the `AdjacencyList`.
    pub fn transition_matrix(&self) -> ndarray::Array2<f32> {
        let n = self.inner.len();

        // Compute the (flattened) transition matrix.
        let mut transition_matrix = vec![0_f32; n * n];
        for (i, (_, neighbors)) in self.inner.iter().enumerate() {
            for (j, (_, d)) in neighbors.iter().enumerate() {
                transition_matrix[i * n + j] = d.as_f32().recip();
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

    /// Iterate over the `Cluster`s in the `AdjacencyList`.
    pub fn clusters(&self) -> Vec<&'a C> {
        self.inner.keys().copied().collect()
    }

    /// Iterate over the edges in the `AdjacencyList`.
    pub fn iter_edges(&self) -> impl Iterator<Item = (&C, &C, U)> + '_ {
        self.inner
            .iter()
            .flat_map(|(&u, neighbors)| neighbors.iter().map(move |(&v, d)| (u, v, *d)))
    }
}

impl<'a, I: Send + Sync, U: Number, D: ParDataset<I, U>, C: ParCluster<I, U, D>> AdjacencyList<'a, I, U, D, C> {
    /// Parallel version of `new`.
    pub fn par_new(clusters: &[&'a C], data: &D) -> Vec<Self> {
        // TODO: This is a naive implementation of creating an adjacency list.
        // We can improve this by using the search functionality from CAKES.
        let inner = clusters
            .par_iter()
            .enumerate()
            .map(|(i, &u)| {
                let neighbors = clusters
                    .par_iter()
                    .enumerate()
                    .filter(|&(j, _)| i != j)
                    .filter_map(|(_, &v)| {
                        let (ru, rv) = (u.radius(), v.radius());
                        let d = u.distance_to_other(data, v);
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

        let c = Self {
            inner,
            _phantom: std::marker::PhantomData,
        };

        let [mut c, mut other] = c.partition();
        let mut components = vec![c];
        while !other.inner.is_empty() {
            [c, other] = other.partition();
            components.push(c);
        }

        components
    }
}
