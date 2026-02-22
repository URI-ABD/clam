//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::cmp::Reverse;

use rayon::prelude::*;

use crate::{
    Dataset, DistanceValue, Tree,
    cakes::{approximate::KnnDfs, d_max, d_min},
    pancakes::{Codec, MaybeCompressedItem},
    utils::SizedHeap,
};

use super::super::{CompressiveSearch, ParCompressiveSearch, leaf_into_hits, par_leaf_into_hits, par_pop_till_leaf, pop_till_leaf};

impl<Item, D, M, T, A> CompressiveSearch<Item, D, M, T, A> for KnnDfs
where
    Item: Codec,
    D: Dataset<Item = MaybeCompressedItem<Item>>,
    T: DistanceValue,
    M: Fn(&Item, &Item) -> T,
{
    fn compressive_search(&self, tree: &mut Tree<D, M, T, A>, query: &Item) -> Result<Vec<(usize, T)>, String> {
        if self.k > tree.dataset.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            tree.decompress_subtree(0)?;
            return tree
                .dataset
                .as_slice()
                .iter()
                .enumerate()
                .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                .collect();
        }
        // let tol = 0.01; // Tolerance for hit improvement.

        let radius = tree.root().radius();
        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None); // (cluster_id, Reverse((d_min, d_max, d)))
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k)); // (item_id, distance)

        let d = tree.dataset.as_slice()[0].1.distance_to_query(query, &tree.metric)?;
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

impl<Item, D, M, T, A> ParCompressiveSearch<Item, D, M, T, A> for KnnDfs
where
    Item: Codec + Send + Sync,
    Item::Compressed: Send + Sync,
    D: Dataset<Item = MaybeCompressedItem<Item>> + Send + Sync,
    D::Id: Send + Sync,
    M: Fn(&Item, &Item) -> T + Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
{
    fn par_compressive_search(&self, tree: &mut Tree<D, M, T, A>, query: &Item) -> Result<Vec<(usize, T)>, String> {
        if self.k > tree.dataset.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            tree.par_decompress_subtree(0)?;
            return tree
                .dataset
                .as_slice()
                .par_iter()
                .enumerate()
                .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                .collect();
        }
        // let tol = 0.01; // Tolerance for hit improvement.

        let radius = tree.root().radius();
        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None); // (cluster_id, Reverse((d_min, d_max, d)))
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k)); // (item_id, distance)

        let d = tree.dataset.as_slice()[0].1.distance_to_query(query, &tree.metric)?;
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
