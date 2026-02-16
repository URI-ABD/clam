//! A middle-ground between the DFS and BFS algorithms.

use core::borrow::Borrow;

use crate::{Cluster, DistanceValue, NamedAlgorithm, Tree, utils::SizedHeap};

use super::super::{Cakes, KnnLinear, Search, knn_bfs::BfsCandidate};

/// A middle-ground between the DFS and BFS algorithms.
#[must_use]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct KnnSieve {
    /// The number of nearest neighbors to search for.
    pub(crate) k: usize,
}

impl KnnSieve {
    /// Creates a new `KnnSieve` object with the given `k`.
    pub const fn new(k: usize) -> Self {
        Self { k }
    }

    /// Returns a `KnnLinear` object with the same `k`.
    pub const fn linear_variant(self) -> KnnLinear {
        KnnLinear { k: self.k }
    }
}

impl_named_algorithm_for_exact_knn!(KnnSieve, "knn-sieve", r"^knn-sieve::k=(\d+)$");

impl<T: DistanceValue> From<KnnSieve> for Cakes<T> {
    fn from(alg: KnnSieve) -> Self {
        Self::KnnSieve(alg)
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnSieve {
    fn search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)> {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, just run linear search.
            return self.linear_variant().search(tree, query);
        }

        let mut outer;
        let mut next_candidates;
        let mut candidates = Vec::new();
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k));

        {
            profi::prof!("KnnSieve::initialization");
            let root = tree.root();
            let d = (tree.metric)(query.borrow(), tree.items[0].1.borrow());
            hits.push((0, d));
            candidates.push(BfsCandidate::from_cluster(root, d));
        }

        while !candidates.is_empty() {
            next_candidates = Vec::new();

            let max_h = *hits.peek().unwrap_or_else(|| unreachable!("hits is not empty.")).1;
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
            // If k is greater than the number of points in the tree, just run linear search.
            return self.linear_variant().par_search(tree, query);
        }

        self.search(tree, query)
    }
}

/// Some stuff for helping profiling.
#[allow(dead_code)]
fn summarize_candidates<T: DistanceValue>(step: usize, when: &str, candidates: &[BfsCandidate<T>], threshold: T) {
    let [min_d_min, max_d_min, min_d, max_d, min_d_max, max_d_max] = candidates.iter().fold(
        [T::max_value(), T::min_value(), T::max_value(), T::min_value(), T::max_value(), T::min_value()],
        |[min_d_min, max_d_min, min_d, max_d, min_d_max, max_d_max], c| {
            [
                if min_d_min < c.d_min { min_d_min } else { c.d_min },
                if max_d_min > c.d_min { max_d_min } else { c.d_min },
                if min_d < c.d { min_d } else { c.d },
                if max_d > c.d { max_d } else { c.d },
                if min_d_max < c.d_max { min_d_max } else { c.d_max },
                if max_d_max > c.d_max { max_d_max } else { c.d_max },
            ]
        },
    );

    let d_min = (max_d_min - min_d_min) / threshold;
    let d = (max_d - min_d) / threshold;
    let d_max = (max_d_max - min_d_max) / threshold;

    let min_max = d_max - d_min;

    println!(
        "Step {step:04}: {when} d_min = {d_min:04.4}, d = {d:04.4}, d_max = {d_max:04.4}, delta: {min_max:04.4}, n_candidates = {:04}, threshold = {threshold:04.4}",
        candidates.len()
    );
}
