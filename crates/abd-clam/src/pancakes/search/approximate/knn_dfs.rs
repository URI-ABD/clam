//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::{borrow::Borrow, cmp::Reverse};

use crate::{
    DistanceValue,
    cakes::{approximate::KnnDfs, d_max, d_min},
    utils::SizedHeap,
};

use super::super::{Codec, Compressible, CompressiveSearch, PancakesTree, leaf_into_hits, par_leaf_into_hits, par_pop_till_leaf, pop_till_leaf};

impl<Id, I, T, A, M, C> CompressiveSearch<Id, I, T, A, M, C> for KnnDfs
where
    I: Compressible,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    C: Codec<I>,
{
    fn compressive_search<Query: Borrow<I>>(&self, tree: &mut PancakesTree<Id, I, T, A, M, C>, query: &Query) -> Vec<(usize, T)> {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, just run linear search.
            return self.linear_variant().compressive_search(tree, query);
        }
        // let tol = 0.01; // Tolerance for hit improvement.

        let radius = tree.root().radius();
        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None); // (cluster_id, Reverse((d_min, d_max, d)))
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k)); // (item_id, distance)

        let d = tree.distance_to_uncompressed(query, 0);
        hits.push((0, d));
        candidates.push((0, Reverse((d_min(radius, d), d_max(radius, d), d))));

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf_id, d) = pop_till_leaf(query, tree, &mut candidates, &mut hits);
            // Process the leaf and update hits.
            leaf_into_hits(query, tree, &mut hits, leaf_id, d);

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);

            if hits.is_full() && (max_h < min_c || !self.should_stop(min_c, max_h)) {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }
        }

        hits.take_items().collect()
    }

    fn par_compressive_search<Query: Borrow<I> + Send + Sync>(&self, tree: &mut PancakesTree<Id, I, T, A, M, C>, query: &Query) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        I::Compressed: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
        C: Send + Sync,
    {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, just run linear search.
            return self.linear_variant().par_compressive_search(tree, query);
        }

        let radius = tree.root().radius();
        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None); // (cluster_id, Reverse((d_min, d_max, d)))
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k)); // (item_id, distance)

        let d = tree.distance_to_uncompressed(query, 0);
        hits.push((0, d));
        candidates.push((0, Reverse((d_min(radius, d), d_max(radius, d), d))));

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf_id, d) = par_pop_till_leaf(query, tree, &mut candidates, &mut hits);
            // Process the leaf and update hits.
            par_leaf_into_hits(query, tree, &mut hits, leaf_id, d);

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);

            if hits.is_full() && (max_h < min_c || !self.should_stop(min_c, max_h)) {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }
        }

        hits.take_items().collect()
    }
}
