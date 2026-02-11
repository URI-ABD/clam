//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::cmp::Reverse;

use rayon::prelude::*;

use crate::{Cluster, DistanceValue, Tree, utils::SizedHeap};

use super::super::{ParSearch, Search, d_max, d_min, leaf_into_hits, par_leaf_into_hits, par_pop_till_leaf, pop_till_leaf};

/// K-Nearest Neighbor (KNN) search using the Depth-First Sieve algorithm.
///
/// The fields are:
///   1. `k`: The number of nearest neighbors to find.
///   2. `max_leaves`: The maximum number of leaf clusters to visit (`usize::MAX` for no limit).
///   3. `max_dist_comps`: The maximum number of distance computations to perform (`usize::MAX` for no limit).
///
/// If both `max_leaves` and `max_dist_comps` are set to `usize::MAX`, the search is exact and will have the same asymptotic behavior as the
/// [`exact variant`](crate::cakes::KnnDfs) of this algorithm.
#[must_use]
pub struct KnnDfs {
    /// The number of nearest neighbors to find.
    pub(crate) k: usize,
    /// The maximum number of leaf clusters to visit (`usize::MAX` for no limit).
    pub(crate) max_leaves: usize,
    /// The maximum number of distance computations to perform (`usize::MAX` for no limit).
    pub(crate) max_dist_comps: usize,
}

impl KnnDfs {
    /// Creates a new `KnnDfs` search object with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of nearest neighbors to find.
    /// * `max_leaves` - The maximum number of leaf clusters to visit (`usize::MAX` for no limit).
    /// * `max_dist_comps` - The maximum number of distance computations to perform (`usize::MAX` for no limit).
    pub const fn new(k: usize, max_leaves: usize, max_dist_comps: usize) -> Self {
        Self { k, max_leaves, max_dist_comps }
    }

    /// Checks whether we should continue the search.
    pub(crate) const fn should_continue(&self, leaves_visited: usize, distance_computations: usize) -> bool {
        leaves_visited < self.max_leaves && distance_computations < self.max_dist_comps
    }
}

impl core::fmt::Display for KnnDfs {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.max_leaves == usize::MAX && self.max_dist_comps == usize::MAX {
            write!(f, "KnnDfs(k={})", self.k)
        } else if self.max_dist_comps == usize::MAX {
            write!(f, "ApproxKnnDfs(k={},leaves<{})", self.k, self.max_leaves)
        } else if self.max_leaves == usize::MAX {
            write!(f, "ApproxKnnDfs(k={},dist_comps<{})", self.k, self.max_dist_comps)
        } else {
            write!(f, "ApproxKnnDfs(k={},leaves<{},dist_comps<{})", self.k, self.max_leaves, self.max_dist_comps)
        }
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnDfs {
    fn name(&self) -> String {
        format!("{self}")
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let radius = root.radius();

        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            return tree.items.iter().enumerate().map(|(i, (_, item))| (i, (tree.metric())(query, item))).collect();
        }
        // let tol = 0.01; // Tolerance for hit improvement.

        let mut candidates = SizedHeap::<&Cluster<T, A>, Reverse<(T, T, T)>>::new(None);
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k));
        let d = (tree.metric)(query, &tree.items[0].1);
        hits.push((0, d));
        candidates.push((root, Reverse((d_min(radius, d), d_max(radius, d), d))));

        let mut leaves_visited = 0;
        let mut distance_computations = 1;

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf, d, n) = pop_till_leaf(query, tree, &mut candidates, &mut hits);
            leaves_visited += 1;
            distance_computations += n;

            // Process the leaf and update hits.
            distance_computations += leaf_into_hits(query, tree, &mut hits, leaf, d);

            // Get the distances to the farthest hit and the closest possible candidate.
            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);

            if hits.is_full() && (max_h < min_c || !self.should_continue(leaves_visited, distance_computations)) {
                // The closest candidate cannot improve our hits OR we have reached the limit of the allowed resources.
                break;
            }
        }

        hits.take_items().collect()
    }
}

impl<Id, I, T, A, M> ParSearch<Id, I, T, A, M> for KnnDfs
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let radius = root.radius();

        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            return tree
                .items
                .par_iter()
                .enumerate()
                .map(|(i, (_, item))| (i, (tree.metric())(query, item)))
                .collect();
        }
        // let tol = 0.01; // Tolerance for hit improvement.

        let mut candidates = SizedHeap::<&Cluster<T, A>, Reverse<(T, T, T)>>::new(None);
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k));
        let d = (tree.metric)(query, &tree.items[0].1);
        hits.push((0, d));
        candidates.push((root, Reverse((d_min(radius, d), d_max(radius, d), d))));

        let mut leaves_visited = 0;
        let mut distance_computations = 1;

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf, d, n) = par_pop_till_leaf(query, tree, &mut candidates, &mut hits);
            leaves_visited += 1;
            distance_computations += n;

            // Process the leaf and update hits.
            distance_computations += par_leaf_into_hits(query, tree, &mut hits, leaf, d);

            // Get the distances to the farthest hit and the closest possible candidate.
            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);

            if hits.is_full() && (max_h < min_c || !self.should_continue(leaves_visited, distance_computations)) {
                // The closest candidate cannot improve our hits OR we have reached the limit of the allowed resources.
                break;
            }
        }

        hits.take_items().collect()
    }
}
