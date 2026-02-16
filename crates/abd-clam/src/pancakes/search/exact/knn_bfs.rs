//! K-Nearest Neighbors (KNN) search using the Breadth-First Sieve algorithm.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{
    DistanceValue,
    cakes::{KnnBfs, d_max},
    utils::SizedHeap,
};

use super::super::{Codec, Compressible, CompressiveSearch, PancakesTree};

impl<Id, I, T, A, M, C> CompressiveSearch<Id, I, T, A, M, C> for KnnBfs
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

        let mut candidates: Vec<(usize, T, usize, T)> = Vec::new(); // (Cluster center index, distance to cluster center, guaranteed cardinality, radius)
        let root = tree.root();
        candidates.push((0, d_max(root.radius(), d), root.cardinality(), root.radius()));

        while !candidates.is_empty() {
            let mut next_candidates: Vec<(usize, T, usize, T)> = Vec::new();
            candidates = filter_candidates(candidates, self.k);

            for (id, d, car, _) in candidates {
                if (
                    next_candidates.len() <= self.k  // We still need more points to satisfy k, AND
                    && (car < (self.k - next_candidates.len()))  // The cluster cannot provide enough points to get to k
                )  // OR
                || tree.get_cluster_unchecked(id).is_leaf()
                {
                    profi::prof!("KnnBfs::process_leaf");
                    // The cluster is a leaf, so we have to look at its points
                    tree.decompress_subtree(id);

                    let leaf = tree.get_cluster_unchecked(id);
                    if leaf.is_singleton() {
                        // It's a singleton, so just add non-center items with the precomputed distance
                        hits.extend(leaf.subtree_range().map(|i| (i, d)));
                    } else {
                        // Not a singleton, so compute distances to all non-center items and add them to hits
                        let distances = leaf
                            .subtree_range()
                            .zip(tree.items[leaf.subtree_range()].iter())
                            .map(|(i, (_, item, _))| item.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (i, d)))
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap_or_else(|_| unreachable!("We just decompressed the subtree."));
                        hits.extend(distances);
                    }
                } else {
                    profi::prof!("KnnBfs::process_parent");
                    if let Some(child_center_indices) = tree.decompress_child_centers(id) {
                        for (cid, d) in child_center_indices
                            .into_iter()
                            .map(|cid| tree.items[cid].1.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (cid, d)))
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap_or_else(|_| unreachable!("We just decompressed the child centers."))
                        {
                            let c = tree.get_cluster_unchecked(cid);
                            let (car, radius) = (c.cardinality, c.radius);
                            hits.push((cid, d));
                            next_candidates.push((cid, d_max(radius, d), car, radius));
                        }
                    }
                }
            }

            candidates = next_candidates;
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

        let mut candidates: Vec<(usize, T, usize, T)> = Vec::new(); // (Cluster center index, distance to cluster center, guaranteed cardinality, radius)
        let root = tree.root();
        candidates.push((0, d_max(root.radius(), d), root.cardinality(), root.radius()));

        while !candidates.is_empty() {
            let mut next_candidates: Vec<(usize, T, usize, T)> = Vec::new();
            candidates = filter_candidates(candidates, self.k);

            for (id, d, car, _) in candidates {
                if (
                    next_candidates.len() <= self.k  // We still need more points to satisfy k, AND
                    && (car < (self.k - next_candidates.len()))  // The cluster cannot provide enough points to get to k
                )  // OR
                || tree.get_cluster_unchecked(id).is_leaf()
                {
                    profi::prof!("KnnBfs::par_process_leaf");
                    // The cluster is a leaf, so we have to look at its points
                    tree.par_decompress_subtree(id);

                    let leaf = tree.get_cluster_unchecked(id);
                    if leaf.is_singleton() {
                        // It's a singleton, so just add non-center items with the precomputed distance
                        hits.extend(leaf.subtree_range().map(|i| (i, d)));
                    } else {
                        // Not a singleton, so compute distances to all non-center items and add them to hits
                        let distances = leaf
                            .subtree_range()
                            .into_par_iter()
                            .zip(tree.items[leaf.subtree_range()].par_iter())
                            .map(|(i, (_, item, _))| item.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (i, d)))
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap_or_else(|_| unreachable!("We just decompressed the subtree."));
                        hits.extend(distances);
                    }
                } else {
                    profi::prof!("KnnBfs::par_process_parent");
                    if let Some(child_center_indices) = tree.par_decompress_child_centers(id) {
                        for (cid, d) in child_center_indices
                            .into_par_iter()
                            .map(|cid| tree.items[cid].1.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (cid, d)))
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap_or_else(|_| unreachable!("We just decompressed the child centers."))
                        {
                            let c = tree.get_cluster_unchecked(cid);
                            let (car, radius) = (c.cardinality, c.radius);
                            hits.push((cid, d));
                            next_candidates.push((cid, d_max(radius, d), car, radius));
                        }
                    }
                }
            }

            candidates = next_candidates;
        }

        hits.take_items().collect()
    }
}

/// Returns those candidates that are needed to guarantee the k-nearest neighbors.
fn filter_candidates<T: DistanceValue>(mut candidates: Vec<(usize, T, usize, T)>, k: usize) -> Vec<(usize, T, usize, T)> {
    profi::prof!("KnnBfs::filter_candidates");

    let threshold_index = quick_partition(&mut candidates, k);
    let threshold = candidates[threshold_index].1;

    candidates
        .into_iter()
        .filter_map(|(id, d, car, r)| {
            let diam = r + r;
            let d_min = if d <= diam { T::zero() } else { d - diam };
            if d_min <= threshold { Some((id, d, car, r)) } else { None }
        })
        .collect()
}

/// The Quick Partition algorithm, which is a variant of the Quick Select algorithm.
///
/// It finds the k-th smallest element in a list of elements, while also reordering the list so that all elements to the left of the k-th smallest element are
/// less than or equal to it, and all elements to the right of the k-th smallest element are greater than or equal to it.
fn quick_partition<T: DistanceValue>(items: &mut [(usize, T, usize, T)], k: usize) -> usize {
    profi::prof!("KnnBfs::quick_partition");

    qps(items, k, 0, items.len() - 1)
}

/// The recursive helper function for the Quick Partition algorithm.
fn qps<T: DistanceValue>(items: &mut [(usize, T, usize, T)], k: usize, l: usize, r: usize) -> usize {
    if l >= r {
        core::cmp::min(l, r)
    } else {
        // Choose the pivot point
        let pivot = l + (r - l) / 2;
        let p = find_pivot(items, l, r, pivot);

        // Calculate the cumulative guaranteed cardinalities for the first p `Cluster`s
        let cumulative_guarantees = items
            .iter()
            .take(p)
            .scan(0, |acc, &(_, _, car, _)| {
                *acc += car;
                Some(*acc)
            })
            .collect::<Vec<_>>();

        // Calculate the guaranteed cardinality of the p-th `Cluster`
        let guaranteed_p = if p > 0 { cumulative_guarantees[p - 1] } else { 0 };

        match guaranteed_p.cmp(&k) {
            core::cmp::Ordering::Equal => p,                      // Found the k-th smallest element
            core::cmp::Ordering::Less => qps(items, k, p + 1, r), // Need to look to the right
            core::cmp::Ordering::Greater => {
                // The `Cluster` just before the p-th might be the one we need
                let guaranteed_p_minus_one = if p > 1 { cumulative_guarantees[p - 2] } else { 0 };
                if p == 0 || guaranteed_p_minus_one < k {
                    p // Found the k-th smallest element
                } else {
                    // Need to look to the left
                    qps(items, k, l, p - 1)
                }
            }
        }
    }
}

/// Moves pivot point and swaps elements around so that all elements to left of pivot are less than or equal to pivot and all elements to right of pivot are
/// greater than pivot.
fn find_pivot<T: DistanceValue>(items: &mut [(usize, T, usize, T)], l: usize, r: usize, pivot: usize) -> usize {
    profi::prof!("KnnBfs::find_pivot");

    // Move pivot to the end
    items.swap(pivot, r);

    // Partition around pivot
    let (mut a, mut b) = (l, l);
    // Invariant: a <= b <= r
    while b < r {
        // If the current element is less than the pivot, swap it with the element at a and increment a.
        if items[b].1 < items[r].1 {
            items.swap(a, b);
            a += 1;
        }
        // Increment b
        b += 1;
    }

    // Move pivot to its final position
    items.swap(a, r);

    a
}
