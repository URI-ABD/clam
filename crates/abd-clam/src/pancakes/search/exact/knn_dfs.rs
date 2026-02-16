//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::{borrow::Borrow, cmp::Reverse};

use rayon::prelude::*;

use crate::{
    DistanceValue,
    cakes::{KnnDfs, d_max, d_min},
    utils::SizedHeap,
};

use super::super::{Codec, Compressible, CompressiveSearch, PancakesTree};

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

        let mut hits = SizedHeap::<usize, T>::new(Some(self.k));
        let d = tree.distance_to_uncompressed(query, 0);
        hits.push((0, d));

        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None); // (cluster_id, Reverse((d_min, d_max, d)))
        let radius = tree.root().radius();
        candidates.push((0, Reverse((d_min(radius, d), d_max(radius, d), d))));

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf_id, d) = pop_till_leaf(query, tree, &mut candidates, &mut hits);
            // Process the leaf and update hits.
            leaf_into_hits(query, tree, &mut hits, leaf_id, d);

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);
            if hits.is_full() && max_h < min_c {
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

        let mut hits = SizedHeap::<usize, T>::new(Some(self.k));
        let d = tree.distance_to_uncompressed(query, 0);
        hits.push((0, d));

        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None); // (cluster_id, Reverse((d_min, d_max, d)))
        let radius = tree.root().radius();
        candidates.push((0, Reverse((d_min(radius, d), d_max(radius, d), d))));

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf_id, d) = par_pop_till_leaf(query, tree, &mut candidates, &mut hits);
            // Process the leaf and update hits.
            par_leaf_into_hits(query, tree, &mut hits, leaf_id, d);

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);
            if hits.is_full() && max_h < min_c {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }
        }

        hits.take_items().collect()
    }
}

/// Pop candidates until the top candidate is a leaf, then pop and return that leaf along with its minimum distance from the query.
///
/// The user must ensure that `candidates` is non-empty before calling this function.
pub fn pop_till_leaf<Id, I, T, A, M, C, Query>(
    query: &Query,
    tree: &mut PancakesTree<Id, I, T, A, M, C>,
    candidates: &mut SizedHeap<usize, Reverse<(T, T, T)>>,
    hits: &mut SizedHeap<usize, T>,
) -> (usize, T)
where
    I: Compressible,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    C: Codec<I>,
    Query: Borrow<I>,
{
    profi::prof!("KnnDfs::pop_till_leaf");

    while candidates
        .peek()
        .and_then(|(&id, _)| tree.items[id].2.as_cluster())
        .filter(|c| !c.is_leaf())
        .is_some()
    {
        profi::prof!("pop-while-not-leaf");

        if let Some((id, _)) = candidates.pop()
            && let Some(child_center_indices) = tree.decompress_child_centers(id)
        {
            child_center_indices.len();

            let distances = child_center_indices
                .into_iter()
                .map(|i| tree.items[i].1.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (i, d)))
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|_| unreachable!("We just decompressed child centers."));

            for (cid, d) in distances {
                let radius = tree.get_cluster_unchecked(cid).radius();
                hits.push((cid, d));
                candidates.push((cid, Reverse((d_min(radius, d), d_max(radius, d), d))));
            }
        }
    }

    let (leaf, d) = candidates
        .pop()
        .map_or_else(|| unreachable!("`candidates` is non-empty."), |(leaf, Reverse((_, _, d)))| (leaf, d));
    (leaf, d)
}

/// Given a leaf cluster, compute the distance from the query to each item in the leaf and push them onto `hits`.
///
/// Returns the number of distance computations performed, excluding the distance to the center (which is already known).
pub fn leaf_into_hits<Id, I, T, A, M, C, Query>(query: &Query, tree: &mut PancakesTree<Id, I, T, A, M, C>, hits: &mut SizedHeap<usize, T>, leaf_id: usize, d: T)
where
    I: Compressible,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    C: Codec<I>,
    Query: Borrow<I>,
{
    profi::prof!("KnnDfs::leaf_into_hits");

    tree.decompress_subtree(leaf_id);
    let leaf = tree.get_cluster_unchecked(leaf_id);

    if leaf.is_singleton() {
        // A singleton leaf has zero radius, so all items in the leaf are exactly `d` from the query.
        hits.extend(leaf.subtree_range().map(|i| (i, d)));
    } else {
        // A non-singleton leaf may have non-zero radius, so we need to compute the distance from the query to each item in the leaf.
        let distances = leaf
            .subtree_range()
            .zip(tree.items[leaf.subtree_range()].iter())
            .map(|(i, (_, item, _))| item.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (i, d)))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|_| unreachable!("We just decompressed the leaf."));
        hits.extend(distances);
    }
}

/// Parallel version of [`pop_till_leaf`].
pub fn par_pop_till_leaf<Id, I, T, A, M, C, Query>(
    query: &Query,
    tree: &mut PancakesTree<Id, I, T, A, M, C>,
    candidates: &mut SizedHeap<usize, Reverse<(T, T, T)>>,
    hits: &mut SizedHeap<usize, T>,
) -> (usize, T)
where
    Id: Send + Sync,
    I: Compressible + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    C: Codec<I> + Send + Sync,
    Query: Borrow<I> + Send + Sync,
{
    profi::prof!("KnnDfs::pop_till_leaf");

    while candidates
        .peek()
        .and_then(|(&id, _)| tree.items[id].2.as_cluster())
        .filter(|c| !c.is_leaf())
        .is_some()
    {
        profi::prof!("pop-while-not-leaf");

        if let Some((id, _)) = candidates.pop()
            && let Some(child_center_indices) = tree.par_decompress_child_centers(id)
        {
            let distances = child_center_indices
                .into_par_iter()
                .map(|i| tree.items[i].1.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (i, d)))
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|_| unreachable!("We just decompressed child centers."));

            for (cid, d) in distances {
                let radius = tree.get_cluster_unchecked(cid).radius();
                hits.push((cid, d));
                candidates.push((cid, Reverse((d_min(radius, d), d_max(radius, d), d))));
            }
        }
    }

    let (leaf, d) = candidates
        .pop()
        .map_or_else(|| unreachable!("`candidates` is non-empty."), |(leaf, Reverse((_, _, d)))| (leaf, d));
    (leaf, d)
}

/// Parallel version of [`leaf_into_hits`].
pub fn par_leaf_into_hits<Id, I, T, A, M, C, Query>(
    query: &Query,
    tree: &mut PancakesTree<Id, I, T, A, M, C>,
    hits: &mut SizedHeap<usize, T>,
    leaf_id: usize,
    d: T,
) where
    Id: Send + Sync,
    I: Compressible + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    C: Codec<I> + Send + Sync,
    Query: Borrow<I> + Send + Sync,
{
    profi::prof!("KnnDfs::leaf_into_hits");

    tree.par_decompress_subtree(leaf_id);
    let leaf = tree.get_cluster_unchecked(leaf_id);

    if leaf.is_singleton() {
        // A singleton leaf has zero radius, so all items in the leaf are exactly `d` from the query.
        hits.extend(leaf.subtree_range().map(|i| (i, d)));
    } else {
        // A non-singleton leaf may have non-zero radius, so we need to compute the distance from the query to each item in the leaf.
        let distances = leaf
            .subtree_range()
            .into_par_iter()
            .zip(tree.items[leaf.subtree_range()].par_iter())
            .map(|(i, (_, item, _))| item.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (i, d)))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|_| unreachable!("We just decompressed the leaf."));
        hits.extend(distances);
    }
}
