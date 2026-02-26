//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::cmp::Reverse;

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    cakes::{KnnDfs, d_max, d_min},
    pancakes::{Codec, MaybeCompressed},
    utils::SizedHeap,
};

use super::super::{CompressiveSearch, ParCompressiveSearch};

impl<Id, I, T, A, M> CompressiveSearch<Id, I, T, A, M> for KnnDfs
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        if self.0 > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            tree.decompress_subtree(0);
            return tree
                .items
                .iter()
                .enumerate()
                .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                .collect();
        }

        let radius = tree.root().radius();
        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None); // (cluster_id, Reverse((d_min, d_max, d)))
        let mut hits = SizedHeap::<usize, T>::new(Some(self.0));

        let d = tree.items[0].1.distance_to_query(query, &tree.metric)?;
        hits.push((0, d));
        candidates.push((0, Reverse((d_min(radius, d), d_max(radius, d), d))));

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf, d, _) = pop_till_leaf(query, tree, &mut candidates, &mut hits)?;
            // Process the leaf and update hits.
            leaf_into_hits(query, tree, &mut hits, leaf, d)?;

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);
            if hits.is_full() && max_h < min_c {
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
        if self.0 > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            tree.decompress_subtree(0);
            return tree
                .items
                .par_iter()
                .enumerate()
                .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                .collect();
        }

        let radius = tree.root().radius();
        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None); // (cluster_id, Reverse((d_min, d_max, d)))
        let mut hits = SizedHeap::<usize, T>::new(Some(self.0));

        let d = tree.items[0].1.distance_to_query(query, &tree.metric)?;
        hits.push((0, d));
        candidates.push((0, Reverse((d_min(radius, d), d_max(radius, d), d))));

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf, d, _) = par_pop_till_leaf(query, tree, &mut candidates, &mut hits)?;
            // Process the leaf and update hits.
            par_leaf_into_hits(query, tree, &mut hits, leaf, d)?;

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates.peek().map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);
            if hits.is_full() && max_h < min_c {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }
        }

        Ok(hits.take_items().collect())
    }
}

/// Pop candidates until the top candidate is a leaf, then pop and return that leaf along with its minimum distance from the query.
///
/// The user must ensure that `candidates` is non-empty before calling this function.
pub fn pop_till_leaf<Id, I, T, A, M>(
    query: &I,
    tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>,
    candidates: &mut SizedHeap<usize, Reverse<(T, T, T)>>,
    hits: &mut SizedHeap<usize, T>,
) -> Result<(usize, T, usize), String>
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    profi::prof!("KnnDfs::pop_till_leaf");

    let mut distance_computations = 0;

    while candidates
        .peek()
        .and_then(|(id, _)| tree.cluster_map.get(id))
        .filter(|c| !c.is_leaf())
        .is_some()
    {
        profi::prof!("pop-while-not-leaf");

        if let Some((id, _)) = candidates.pop()
            && let Ok(child_center_indices) = tree.decompress_child_centers(id)
        {
            distance_computations += child_center_indices.len();

            let distances = child_center_indices
                .into_iter()
                .map(|i| tree.items[i].1.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                .collect::<Result<Vec<_>, _>>()?;

            for (cid, d) in distances {
                let child = tree.get_cluster(cid)?;
                let radius = child.radius();
                hits.push((cid, d));
                candidates.push((cid, Reverse((d_min(radius, d), d_max(radius, d), d))));
            }
        }
    }

    let (leaf, d) = candidates
        .pop()
        .map_or_else(|| unreachable!("`candidates` is non-empty."), |(leaf, Reverse((_, _, d)))| (leaf, d));
    Ok((leaf, d, distance_computations))
}

/// Given a leaf cluster, compute the distance from the query to each item in the leaf and push them onto `hits`.
///
/// Returns the number of distance computations performed, excluding the distance to the center (which is already known).
pub fn leaf_into_hits<Id, I, T, A, M>(
    query: &I,
    tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>,
    hits: &mut SizedHeap<usize, T>,
    leaf_id: usize,
    d: T,
) -> Result<usize, String>
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    profi::prof!("KnnDfs::leaf_into_hits");

    tree.decompress_subtree(leaf_id);
    let leaf = tree.get_cluster(leaf_id)?;

    if leaf.is_singleton() {
        // A singleton leaf has zero radius, so all items in the leaf are exactly `d` from the query.
        hits.extend(leaf.subtree_indices().map(|i| (i, d)));
        Ok(0)
    } else {
        // A non-singleton leaf may have non-zero radius, so we need to compute the distance from the query to each item in the leaf.
        let distances = leaf
            .subtree_indices()
            .zip(tree.items[leaf.subtree_indices()].iter())
            .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
            .collect::<Result<Vec<_>, _>>()?;
        hits.extend(distances);
        Ok(leaf.cardinality() - 1) // We already knew the distance to the center.
    }
}

/// Parallel version of [`pop_till_leaf`].
pub fn par_pop_till_leaf<Id, I, T, A, M>(
    query: &I,
    tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>,
    candidates: &mut SizedHeap<usize, Reverse<(T, T, T)>>,
    hits: &mut SizedHeap<usize, T>,
) -> Result<(usize, T, usize), String>
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    profi::prof!("KnnDfs::pop_till_leaf");

    let mut distance_computations = 0;

    while candidates
        .peek()
        .and_then(|(id, _)| tree.cluster_map.get(id))
        .filter(|c| !c.is_leaf())
        .is_some()
    {
        profi::prof!("pop-while-not-leaf");

        if let Some((id, _)) = candidates.pop()
            && let Some(child_center_indices) = tree.par_decompress_child_centers(id)?
        {
            distance_computations += child_center_indices.len();

            let distances = child_center_indices
                .into_par_iter()
                .map(|i| tree.items[i].1.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                .collect::<Result<Vec<_>, _>>()?;

            for (cid, d) in distances {
                let child = tree.get_cluster(cid)?;
                let radius = child.radius();
                hits.push((cid, d));
                candidates.push((cid, Reverse((d_min(radius, d), d_max(radius, d), d))));
            }
        }
    }

    let (leaf, d) = candidates
        .pop()
        .map_or_else(|| unreachable!("`candidates` is non-empty."), |(leaf, Reverse((_, _, d)))| (leaf, d));
    Ok((leaf, d, distance_computations))
}

/// Parallel version of [`leaf_into_hits`].
pub fn par_leaf_into_hits<Id, I, T, A, M>(
    query: &I,
    tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>,
    hits: &mut SizedHeap<usize, T>,
    leaf_id: usize,
    d: T,
) -> Result<usize, String>
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    profi::prof!("KnnDfs::leaf_into_hits");

    tree.par_decompress_subtree(leaf_id);
    let leaf = tree.get_cluster(leaf_id)?;

    if leaf.is_singleton() {
        // A singleton leaf has zero radius, so all items in the leaf are exactly `d` from the query.
        hits.extend(leaf.subtree_indices().map(|i| (i, d)));
        Ok(0)
    } else {
        // A non-singleton leaf may have non-zero radius, so we need to compute the distance from the query to each item in the leaf.
        let distances = leaf
            .subtree_indices()
            .into_par_iter()
            .zip(tree.items[leaf.subtree_indices()].par_iter())
            .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
            .collect::<Result<Vec<_>, _>>()?;
        hits.extend(distances);
        Ok(leaf.cardinality() - 1) // We already knew the distance to the center.
    }
}
