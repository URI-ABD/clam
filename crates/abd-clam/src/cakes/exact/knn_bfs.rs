//! K-Nearest Neighbors (KNN) search using the Breadth-First Sieve algorithm.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{Cluster, DistanceValue, NamedAlgorithm, Tree, utils::SizedHeap};

use super::super::{Cakes, KnnLinear, Search, d_max, d_min};

/// K-Nearest Neighbor (KNN) search using the Breadth-First Sieve algorithm.
#[must_use]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct KnnBfs {
    /// The number of nearest neighbors to search for.
    pub(crate) k: usize,
}

impl KnnBfs {
    /// Creates a new `KnnBfs` object with the given `k`.
    pub const fn new(k: usize) -> Self {
        Self { k }
    }

    /// Returns a `KnnLinear` object with the same `k`.
    pub const fn linear_variant(self) -> KnnLinear {
        KnnLinear { k: self.k }
    }
}

impl_named_algorithm_for_exact_knn!(KnnBfs, "knn-bfs", r"^knn-bfs::k=(\d+)$");

impl<T: DistanceValue> From<KnnBfs> for Cakes<T> {
    fn from(alg: KnnBfs) -> Self {
        Self::KnnBfs(alg)
    }
}

/// A candidate cluster for the BFS algorithm.
#[derive(Debug, Clone)]
pub struct BfsCandidate<T> {
    /// The index of the cluster's center in the tree's items vector.
    pub id: usize,
    /// The minimum distance from the query to any point in the cluster.
    pub d_min: T,
    /// The distance from the query to the center of the cluster.
    pub d: T,
    /// The maximum distance from the query to any point in the cluster.
    pub d_max: T,
    /// The number of points in the cluster, which is used for determining when we have enough candidates to satisfy k.
    pub cardinality: usize,
    /// Whether the cluster is a leaf cluster (i.e., it has no children).
    pub is_leaf: bool,
    /// Whether the cluster is a singleton (i.e., it contains only one point or has zero radius).
    pub is_singleton: bool,
}

impl<T: DistanceValue> BfsCandidate<T> {
    /// Creates a new `BfsCandidate` from a `Cluster` and the distance from the query to the cluster's center.
    pub fn from_cluster<A>(cluster: &Cluster<T, A>, d: T) -> Self {
        Self {
            id: cluster.center_index,
            d_min: d_min(cluster.radius, d),
            d,
            d_max: d_max(cluster.radius, d),
            cardinality: cluster.cardinality,
            is_leaf: cluster.is_leaf(),
            is_singleton: cluster.is_singleton(),
        }
    }

    /// Get a threshold distance for this candidate, which is used to order candidates and determine which ones to keep when filtering.
    const fn threshold(&self) -> T {
        self.d
    }

    /// Get the indices of the items in the cluster that are not the center.
    pub const fn subtree_indices(&self) -> core::ops::Range<usize> {
        (self.id + 1)..(self.id + self.cardinality)
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnBfs {
    fn search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)> {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, just run linear search.
            return self.linear_variant().search(tree, query);
        }

        let mut next_candidates;
        let mut candidates = Vec::new();
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k));

        {
            profi::prof!("KnnBfs::initialization");
            let root = tree.root();
            let d = (tree.metric)(query.borrow(), tree.items[0].1.borrow());
            hits.push((0, d));
            candidates.push(BfsCandidate::from_cluster(root, d));
        }

        while !candidates.is_empty() {
            next_candidates = Vec::new();
            filter_candidates(&mut candidates, self.k, hits.len());

            for candidate in candidates {
                if candidate.cardinality < self.k {
                    // The candidate cannot provide enough points to satisfy k, so we have to look at all its points
                    profi::prof!("KnnBfs::process_tiny");
                    if candidate.is_singleton {
                        // It's a singleton, so just add non-center items with the precomputed distance
                        hits.extend(candidate.subtree_indices().map(|i| (i, candidate.d)));
                    } else {
                        // Not a singleton, so compute distances to all non-center items and add them to hits
                        let distances = candidate
                            .subtree_indices()
                            .zip(tree.items[candidate.subtree_indices()].iter())
                            .map(|(i, (_, item, _))| (i, (tree.metric)(query.borrow(), item.borrow())));
                        hits.extend(distances);
                    }
                } else if let Some(cids) = tree.items[candidate.id].2.as_cluster().and_then(Cluster::child_center_indices) {
                    // The candidate is a parent cluster so we can look at its children
                    profi::prof!("KnnBfs::process_parent");
                    for (child, d) in cids.iter().filter_map(|&cid| {
                        let (_, item, loc) = &tree.items[cid];
                        loc.as_cluster().map(|child| (child, (tree.metric)(query.borrow(), item.borrow())))
                    }) {
                        hits.push((child.center_index, d));
                        next_candidates.push(BfsCandidate::from_cluster(child, d));
                    }
                } else {
                    // The candidate is a leaf cluster with more points than k, so we have to look at all its points
                    profi::prof!("KnnBfs::process_leaf");
                    if candidate.is_singleton {
                        // It's a singleton, so just add non-center items with the precomputed distance
                        hits.extend(candidate.subtree_indices().map(|i| (i, candidate.d)));
                    } else {
                        // Not a singleton, so compute distances to all non-center items and add them to hits
                        let distances = candidate
                            .subtree_indices()
                            .zip(tree.items[candidate.subtree_indices()].iter())
                            .map(|(i, (_, item, _))| (i, (tree.metric)(query.borrow(), item.borrow())));
                        hits.extend(distances);
                    }
                }
            }

            candidates = next_candidates;
        }

        hits.take_items().collect()
    }

    fn par_search<Item: Borrow<I> + Send + Sync, Query: Borrow<I> + Send + Sync>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, just run linear search.
            return self.linear_variant().par_search(tree, query);
        }

        let mut next_candidates;
        let mut candidates = Vec::new();
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k));

        {
            profi::prof!("KnnBfs::par_initialization");
            let root = tree.root();
            let d = (tree.metric)(query.borrow(), tree.items[0].1.borrow());
            hits.push((0, d));
            candidates.push(BfsCandidate::from_cluster(root, d));
        }

        while !candidates.is_empty() {
            next_candidates = Vec::new();
            filter_candidates(&mut candidates, self.k, hits.len());

            for candidate in candidates {
                if candidate.cardinality < self.k {
                    // The candidate cannot provide enough points to satisfy k, so we have to look at all its points
                    profi::prof!("KnnBfs::par_process_tiny");
                    if candidate.is_singleton {
                        // It's a singleton, so just add non-center items with the precomputed distance
                        hits.extend(candidate.subtree_indices().map(|i| (i, candidate.d)));
                    } else {
                        // Not a singleton, so compute distances to all non-center items and add them to hits
                        let distances = candidate
                            .subtree_indices()
                            .into_par_iter()
                            .zip(tree.items[candidate.subtree_indices()].par_iter())
                            .map(|(i, (_, item, _))| (i, (tree.metric)(query.borrow(), item.borrow())))
                            .collect::<Vec<_>>();
                        hits.extend(distances);
                    }
                } else if let Some(cids) = tree.items[candidate.id].2.as_cluster().and_then(Cluster::child_center_indices) {
                    // The candidate is a parent cluster so we can look at its children
                    profi::prof!("KnnBfs::par_process_parent");
                    for (child, d) in cids
                        .par_iter()
                        .filter_map(|&cid| {
                            let (_, item, loc) = &tree.items[cid];
                            loc.as_cluster().map(|child| (child, (tree.metric)(query.borrow(), item.borrow())))
                        })
                        .collect::<Vec<_>>()
                    {
                        hits.push((child.center_index, d));
                        next_candidates.push(BfsCandidate::from_cluster(child, d));
                    }
                } else {
                    // The candidate is a leaf cluster with more points than k, so we have to look at all its points
                    profi::prof!("KnnBfs::par_process_leaf");
                    if candidate.is_singleton {
                        // It's a singleton, so just add non-center items with the precomputed distance
                        hits.extend(candidate.subtree_indices().map(|i| (i, candidate.d)));
                    } else {
                        // Not a singleton, so compute distances to all non-center items and add them to hits
                        let distances = candidate
                            .subtree_indices()
                            .into_par_iter()
                            .zip(tree.items[candidate.subtree_indices()].par_iter())
                            .map(|(i, (_, item, _))| (i, (tree.metric)(query.borrow(), item.borrow())))
                            .collect::<Vec<_>>();
                        hits.extend(distances);
                    }
                }
            }

            candidates = next_candidates;
        }

        hits.take_items().collect()
    }
}

/// Removes candidates whose minimum distance is greater than the threshold distance, which is the distance of the k-th closest candidate.
///
/// # Returns
///
/// The threshold distance of the k-th closest candidate.
pub fn filter_candidates<T: DistanceValue>(candidates: &mut Vec<BfsCandidate<T>>, k: usize, n_hits: usize) -> T {
    profi::prof!("KnnBfs::filter_candidates");

    let i = quick_partition(candidates, k, n_hits);
    let threshold = candidates[i].threshold();
    candidates.retain(|c| c.d_min <= threshold);
    threshold
}

/// The Quick Partition algorithm, which is a variant of the Quick Select algorithm.
///
/// It finds the k-th smallest element in a list of elements, while also reordering the list so that all elements to the left of the k-th smallest element are
/// less than or equal to it, and all elements to the right of the k-th smallest element are greater than or equal to it.
fn quick_partition<T: DistanceValue>(items: &mut [BfsCandidate<T>], k: usize, n_hits: usize) -> usize {
    qps(items, k, n_hits, 0, items.len() - 1)
}

/// The recursive helper function for the Quick Partition algorithm.
fn qps<T: DistanceValue>(items: &mut [BfsCandidate<T>], k: usize, n_hits: usize, l: usize, r: usize) -> usize {
    if l >= r {
        l.min(r) // Only one element left, which must be the k-th smallest
    } else {
        // Choose the pivot point
        let pivot = l + (r - l) / 2;
        let p = find_pivot(items, l, r, pivot);

        // Calculate the cumulative guaranteed cardinalities for the first p `Cluster`s
        let cumulative_guarantees = {
            items
                .iter()
                .take(p)
                .scan(n_hits, |acc, c| {
                    *acc += c.cardinality;
                    Some(*acc)
                })
                .collect::<Vec<_>>()
        };

        // Calculate the guaranteed cardinality of the p-th `Cluster`
        let guaranteed_p = if p > 0 { cumulative_guarantees[p - 1] } else { 0 };

        match guaranteed_p.cmp(&k) {
            core::cmp::Ordering::Equal => p,                              // Found the k-th smallest element
            core::cmp::Ordering::Less => qps(items, k, n_hits, p + 1, r), // Need to look to the right
            core::cmp::Ordering::Greater => {
                // The `Cluster` just before the p-th might be the one we need
                let guaranteed_p_minus_one = if p > 1 { cumulative_guarantees[p - 2] } else { 0 };
                if p == 0 || guaranteed_p_minus_one < k {
                    p // Found the k-th smallest element
                } else {
                    // Need to look to the left
                    qps(items, k, n_hits, l, p - 1)
                }
            }
        }
    }
}

/// Moves pivot point and swaps elements around so that all elements to left of pivot are less than or equal to pivot and all elements to right of pivot are
/// greater than pivot.
fn find_pivot<T: DistanceValue>(items: &mut [BfsCandidate<T>], l: usize, r: usize, pivot: usize) -> usize {
    // Move pivot to the end
    items.swap(pivot, r);

    // Partition around pivot
    let (mut a, mut b) = (l, l);
    // Invariant: a <= b <= r
    while b < r {
        // If the current element is less than the pivot, swap it with the element at a and increment a.
        if items[b].threshold() < items[r].threshold() {
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
