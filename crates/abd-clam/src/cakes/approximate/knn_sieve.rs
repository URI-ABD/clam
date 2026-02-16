//! Approximate version of the [`KnnSieve`](crate::cakes::KnnSieve) algorithm.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{Cluster, DistanceValue, NamedAlgorithm, Tree, utils::SizedHeap};

use super::super::{Cakes, KnnLinear, Search, knn_bfs::BfsCandidate};

/// Approximate version of the [`KnnSieve`](crate::cakes::KnnSieve) algorithm.
#[must_use]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct KnnSieve {
    /// The number of nearest neighbors to return.
    pub(crate) k: usize,
    /// The approximation tolerance. Higher values allow for more approximation, potentially improving speed at the cost of accuracy.
    pub(crate) tol: f64,
}

impl_named_algorithm_for_approx_knn!(KnnSieve, "approx-knn-sieve", r"^approx-knn-sieve::k=(\d+),tol=([0-9]*\.?[0-9]+)$");

impl<T: DistanceValue> From<KnnSieve> for Cakes<T> {
    fn from(alg: KnnSieve) -> Self {
        Self::ApproxKnnSieve(alg)
    }
}

impl KnnSieve {
    /// Creates a new `KnnSieve` algorithm with the specified number of nearest neighbors and approximation tolerance.
    pub const fn new(k: usize, tol: f64) -> Self {
        Self { k, tol }
    }

    /// Returns a `KnnLinear` object with the same `k`.
    pub const fn linear_variant(self) -> KnnLinear {
        KnnLinear { k: self.k }
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnSieve {
    fn search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)> {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            return tree
                .items
                .iter()
                .enumerate()
                .map(|(i, (_, item, _))| (i, (tree.metric)(query.borrow(), item.borrow())))
                .collect();
        }

        let mut outer;
        let mut next_candidates;
        let mut candidates = Vec::new();
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k));

        let mut prev_max_h;
        let mut ratio_ema = 1.0;
        {
            profi::prof!("KnnSieve::initialization");
            let root = tree.root();
            let d = (tree.metric)(query.borrow(), tree.items[0].1.borrow());
            hits.push((0, d));
            candidates.push(BfsCandidate::from_cluster(root, d));

            prev_max_h = d;
        }

        let alpha = 9_f64 / 11.0;
        while !candidates.is_empty() {
            next_candidates = Vec::new();

            let max_h = *hits.peek().unwrap_or_else(|| unreachable!("hits is not empty.")).1;

            let ratio = if hits.is_full() {
                let max_h_f64 = max_h
                    .to_f64()
                    .unwrap_or_else(|| unreachable!("Distance value should always be convertible to f64"));
                let prev_max_h_f64 = prev_max_h
                    .to_f64()
                    .unwrap_or_else(|| unreachable!("Distance value should always be convertible to f64"));
                1.0 - max_h_f64 / prev_max_h_f64
            } else {
                1.0
            };
            ratio_ema = alpha.mul_add(ratio_ema, ratio);
            prev_max_h = max_h;

            if hits.is_full() && ratio_ema < self.tol {
                break;
            }
            candidates.retain(|c| c.d_min <= max_h);

            if hits.is_full() && candidates.len() > 1 {
                (outer, candidates) = candidates.into_iter().partition(|c| max_h < c.d);
                if candidates.is_empty() {
                    core::mem::swap(&mut candidates, &mut outer);
                }
            } else {
                outer = Vec::new();
            }

            for candidate in candidates {
                if candidate.cardinality < self.k {
                    // The candidate cannot provide enough points to satisfy k, so we have to look at all its points
                    profi::prof!("KnnSieve::process_tiny");
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
                    profi::prof!("KnnSieve::process_parent");
                    for (child, d) in cids.iter().filter_map(|&cid| {
                        let (_, item, loc) = &tree.items[cid];
                        loc.as_cluster().map(|child| (child, (tree.metric)(query.borrow(), item.borrow())))
                    }) {
                        hits.push((child.center_index, d));
                        next_candidates.push(BfsCandidate::from_cluster(child, d));
                    }
                } else {
                    // The candidate is a leaf cluster with more points than k, so we have to look at all its points
                    profi::prof!("KnnSieve::process_leaf");
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

            candidates = outer.into_iter().chain(next_candidates).collect();
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
            // If k is greater than the number of points in the tree, return all items with their distances.
            return tree
                .items
                .par_iter()
                .enumerate()
                .map(|(i, (_, item, _))| (i, (tree.metric)(query.borrow(), item.borrow())))
                .collect();
        }

        self.search(tree, query)
    }
}
