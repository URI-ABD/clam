//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::cmp::Reverse;

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    cakes::{approximate::KnnDfs, d_max, d_min},
    pancakes::{Codec, MaybeCompressed},
    utils::SizedHeap,
};

use super::super::{CompressiveSearch, ParCompressiveSearch, leaf_into_hits, par_leaf_into_hits, par_pop_till_leaf, pop_till_leaf};

impl<Id, I, T, A, M> CompressiveSearch<Id, I, T, A, M> for KnnDfs
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            tree.decompress_subtree(0)?;
            return tree
                .items
                .iter()
                .enumerate()
                .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                .collect();
        }
        // let tol = 0.01; // Tolerance for hit improvement.

        let radius = tree.root().radius();
        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None); // (cluster_id, Reverse((d_min, d_max, d)))
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k)); // (item_id, distance)

        let d = tree.items[0].1.distance_to_query(query, &tree.metric)?;
        hits.push((0, d));
        candidates.push((0, Reverse((d_min(radius, d), d_max(radius, d), d))));

        let mut leaves_visited = 0;
        let mut distance_computations = 1;

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf, d, n) = pop_till_leaf(query, tree, &mut candidates, &mut hits)?;
            leaves_visited += 1;
            distance_computations += n;

            // Process the leaf and update hits.
            distance_computations += leaf_into_hits(query, tree, &mut hits, leaf, d)?;

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);

            if hits.is_full() && (max_h < min_c || !self.should_continue(leaves_visited, distance_computations)) {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }
        }

        Ok(hits.take_items().collect())
    }
}

impl<Id, I, T, A, M> ParCompressiveSearch<Id, I, T, A, M> for KnnDfs
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            tree.par_decompress_subtree(0)?;
            return tree
                .items
                .par_iter()
                .enumerate()
                .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                .collect();
        }
        // let tol = 0.01; // Tolerance for hit improvement.

        let radius = tree.root().radius();
        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None); // (cluster_id, Reverse((d_min, d_max, d)))
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k)); // (item_id, distance)

        let d = tree.items[0].1.distance_to_query(query, &tree.metric)?;
        hits.push((0, d));
        candidates.push((0, Reverse((d_min(radius, d), d_max(radius, d), d))));

        let mut leaves_visited = 0;
        let mut distance_computations = 1;

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf, d, n) = par_pop_till_leaf(query, tree, &mut candidates, &mut hits)?;
            leaves_visited += 1;
            distance_computations += n;

            // Process the leaf and update hits.
            distance_computations += par_leaf_into_hits(query, tree, &mut hits, leaf, d)?;

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);

            if hits.is_full() && (max_h < min_c || !self.should_continue(leaves_visited, distance_computations)) {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }
        }

        Ok(hits.take_items().collect())
    }
}
