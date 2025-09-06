//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::cmp::Reverse;

use crate::{
    Cluster, DistanceValue, Tree,
    cakes::{ParSearch, Search, d_max, d_min, leaf_into_hits, pop_till_leaf},
    utils::SizedHeap,
};

/// K-Nearest Neighbor (KNN) search using the Depth-First Sieve algorithm.
///
/// The fields are:
///   1. `k`: The number of nearest neighbors to find.
///   2. `max_leaves`: The maximum number of leaf clusters to visit (`usize::MAX` for no limit).
///   3. `max_dist_comps`: The maximum number of distance computations to perform (`usize::MAX` for no limit).
///
/// If both `max_leaves` and `max_dist_comps` are set to `usize::MAX`, the search is exact and will have the same asymptotic behavior as the
/// [`exact variant`](crate::cakes::KnnDfs) of this algorithm.
pub struct KnnDfs(pub usize, pub usize, pub usize);

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnDfs {
    fn name(&self) -> String {
        if self.1 == usize::MAX && self.2 == usize::MAX {
            format!("KnnDfs(k={})", self.0)
        } else if self.2 == usize::MAX {
            format!("ApproxKnnDfs(k={},leaves<{})", self.0, self.1)
        } else if self.1 == usize::MAX {
            format!("ApproxKnnDfs(k={},dist_comps<{})", self.0, self.2)
        } else {
            format!("ApproxKnnDfs(k={},leaves<{},dist_comps<{})", self.0, self.1, self.2)
        }
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();

        if self.0 > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all
            // items with their distances.
            return tree.distances_to_items_in_cluster(query, root).collect();
        }
        // let tol = 0.01; // Tolerance for hit improvement.

        let mut candidates = SizedHeap::<&Cluster<T, A>, Reverse<(T, T, T)>>::new(None);
        let mut hits = SizedHeap::<usize, T>::new(Some(self.0));
        let d = tree.distance_to_center(query, root);
        hits.push((root.center_index(), d));
        candidates.push((root, Reverse((d_min(root, d), d_max(root, d), d))));

        let mut leaves_visited = 0;
        let mut distance_computations = 1;

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf, d, n) = pop_till_leaf(query, tree, &mut candidates, &mut hits);
            leaves_visited += 1;
            distance_computations += n;

            // Process the leaf and update hits.
            distance_computations += leaf_into_hits(query, tree, &mut hits, leaf, d);

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);

            if hits.is_full()
                && [
                    max_h < min_c,                   // All remaining candidates are too far to improve hits.
                    leaves_visited >= self.1,        // Stop after visiting this many leaves.
                    distance_computations >= self.2, // Stop after this many distance computations.
                ]
                .iter()
                .any(|&x| x)
            {
                // The closest candidate cannot improve our hits, so we can stop.
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
        // For now, just call the single-threaded search.
        self.search(tree, query)
    }
}
