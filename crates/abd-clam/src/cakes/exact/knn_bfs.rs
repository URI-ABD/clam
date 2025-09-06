//! K-Nearest Neighbors (KNN) search using the Breadth-First Sieve algorithm.

use crate::{
    Cluster, DistanceValue, Tree,
    cakes::{ParSearch, Search, d_max},
    utils::SizedHeap,
};

/// K-Nearest Neighbor (KNN) search using the Breadth-First Sieve algorithm.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnBfs(pub usize);

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnBfs {
    fn name(&self) -> String {
        format!("KnnBfs(k={})", self.0)
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();

        if self.0 > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            return tree.distances_to_items_in_cluster(query, root).collect();
        }

        let mut candidates = Vec::new();
        let mut hits = SizedHeap::<usize, T>::new(Some(self.0));

        let d = tree.distance_to_center(query, root);
        hits.push((root.center_index(), d));
        candidates.push((root, d_max(root, d)));

        while !candidates.is_empty() {
            let mut next_candidates = Vec::new();
            candidates = filter_candidates(candidates, self.0);

            for (cluster, d) in candidates {
                if (
                    next_candidates.len() <= self.0  // We still need more points to satisfy k, AND
                    && (cluster.cardinality() < (self.0 - next_candidates.len()))  // The cluster cannot provide enough points to get to k
                )  // OR
                || cluster.is_leaf()
                {
                    profi::prof!("KnnBfs::process_leaf");
                    // The cluster is a leaf, so we have to look at its points
                    if cluster.is_singleton() {
                        // It's a singleton, so just add non-center items with the precomputed distance
                        hits.extend(cluster.items_indices().skip(1).map(|i| (i, d)));
                    } else {
                        // Not a singleton, so compute distances to all non-center items and add them to hits
                        hits.extend(tree.distances_to_items_in_subtree(query, cluster));
                    }
                } else {
                    profi::prof!("KnnBfs::process_parent");
                    for child in tree.children_of(cluster).unwrap_or_else(|| unreachable!("Cluster is a parent")) {
                        let ci = child.center_index();
                        let d = tree.distance_to(query, ci);
                        hits.push((ci, d));
                        next_candidates.push((child, d_max(child, d)));
                    }
                }
            }

            candidates = next_candidates;
        }

        hits.take_items().collect()
    }
}

impl<Id, I, T, A, M> ParSearch<Id, I, T, A, M> for KnnBfs
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

/// Returns those candidates that are needed to guarantee the k-nearest neighbors.
fn filter_candidates<T: DistanceValue, A>(mut candidates: Vec<(&Cluster<T, A>, T)>, k: usize) -> Vec<(&Cluster<T, A>, T)> {
    profi::prof!("KnnBfs::filter_candidates");

    let threshold_index = quick_partition(&mut candidates, k);
    let threshold = candidates[threshold_index].1;

    candidates
        .into_iter()
        .filter_map(|(cluster, d)| {
            let diam = cluster.radius() + cluster.radius();
            let d_min = if d <= diam { T::zero() } else { d - diam };
            if d_min <= threshold { Some((cluster, d)) } else { None }
        })
        .collect()
}

/// The Quick Partition algorithm, which is a variant of the Quick Select algorithm.
///
/// It finds the k-th smallest element in a list of elements, while also reordering the list so that all elements to the left of the k-th smallest element are
/// less than or equal to it, and all elements to the right of the k-th smallest element are greater than or equal to it.
fn quick_partition<T: DistanceValue, A>(items: &mut [(&Cluster<T, A>, T)], k: usize) -> usize {
    profi::prof!("KnnBfs::quick_partition");

    qps(items, k, 0, items.len() - 1)
}

/// The recursive helper function for the Quick Partition algorithm.
fn qps<T: DistanceValue, A>(items: &mut [(&Cluster<T, A>, T)], k: usize, l: usize, r: usize) -> usize {
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
            .scan(0, |acc, (cluster, _)| {
                *acc += cluster.cardinality();
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
fn find_pivot<T: DistanceValue, A>(items: &mut [(&Cluster<T, A>, T)], l: usize, r: usize, pivot: usize) -> usize {
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
