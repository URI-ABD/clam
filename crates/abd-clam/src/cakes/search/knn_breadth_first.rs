//! K-Nearest Neighbors search using a Breadth First sieve.

use core::cmp::{min, Ordering};

use distances::Number;
use rayon::prelude::*;

use crate::{
    cakes::{ParSearchable, Searchable},
    cluster::ParCluster,
    metric::ParMetric,
    Cluster, Metric, SizedHeap,
};

use super::{ParSearchAlgorithm, SearchAlgorithm};

/// K-Nearest Neighbors search using a Breadth First sieve.
pub struct KnnBreadthFirst(pub usize);

impl<I, T: Number, C: Cluster<T>, M: Metric<I, T>, D: Searchable<I, T, C, M>> SearchAlgorithm<I, T, C, M, D>
    for KnnBreadthFirst
{
    fn name(&self) -> &str {
        "KnnBreadthFirst"
    }

    fn radius(&self) -> Option<T> {
        None
    }

    fn k(&self) -> Option<usize> {
        Some(self.0)
    }

    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        let mut candidates = Vec::new();
        let mut hits = SizedHeap::<(T, usize)>::new(Some(self.0));

        let d = data.query_to_center(metric, query, root);
        candidates.push((d_max(root, d), root));

        while !candidates.is_empty() {
            candidates = filter_candidates(candidates, self.0)
                .into_iter()
                .fold(Vec::new(), |mut acc, (d, c)| {
                    if (c.cardinality() < (self.0 - acc.len())) || c.is_leaf() {
                        if c.is_singleton() {
                            c.indices().into_iter().for_each(|i| hits.push((d, i)));
                        } else {
                            data.query_to_all(metric, query, c).for_each(|(i, d)| hits.push((d, i)));
                        }
                    } else {
                        acc.extend(
                            c.children()
                                .into_iter()
                                .map(|c| (d_max(c, data.query_to_center(metric, query, c)), c)),
                        );
                    }
                    acc
                });
        }

        hits.items().map(|(d, i)| (i, d)).collect()
    }
}

impl<I: Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>, D: ParSearchable<I, T, C, M>>
    ParSearchAlgorithm<I, T, C, M, D> for KnnBreadthFirst
{
    #[inline(never)]
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        let mut candidates = Vec::new();
        let mut hits = SizedHeap::<(T, usize)>::new(Some(self.0));

        let d = data.par_query_to_center(metric, query, root);
        candidates.push((d_max(root, d), root));

        while !candidates.is_empty() {
            candidates = filter_candidates(candidates, self.0)
                .into_iter()
                .fold(Vec::new(), |mut acc, (d, c)| {
                    if (c.cardinality() < (self.0 - acc.len())) || c.is_leaf() {
                        if c.is_singleton() {
                            c.indices().into_iter().for_each(|i| hits.push((d, i)));
                        } else {
                            let distances = data.par_query_to_all(metric, query, c).collect::<Vec<_>>();
                            distances.into_iter().for_each(|(i, d)| hits.push((d, i)));
                        }
                    } else {
                        let distances = c
                            .children()
                            .into_par_iter()
                            .map(|c| (d_max(c, data.par_query_to_center(metric, query, c)), c))
                            .collect::<Vec<_>>();
                        acc.extend(distances);
                    }
                    acc
                });
        }

        hits.items().map(|(d, i)| (i, d)).collect()
    }
}

/// Returns the theoretical maximum distance from the query to a point in the cluster.
fn d_max<T: Number, C: Cluster<T>>(c: &C, d: T) -> T {
    c.radius() + d
}

/// Returns those candidates that are needed to guarantee the k-nearest
/// neighbors.
fn filter_candidates<T: Number, C: Cluster<T>>(mut candidates: Vec<(T, &C)>, k: usize) -> Vec<(T, &C)> {
    let threshold_index = quick_partition(&mut candidates, k);
    let threshold = candidates[threshold_index].0;

    candidates
        .into_iter()
        .filter_map(|(d, c)| {
            let diam = c.radius().double();
            let d_min = if d <= diam { T::ZERO } else { d - diam };
            if d_min <= threshold {
                Some((d, c))
            } else {
                None
            }
        })
        .collect()
}

/// The Quick Partition algorithm, which is a variant of the Quick Select
/// algorithm. It finds the k-th smallest element in a list of elements, while
/// also reordering the list so that all elements to the left of the k-th
/// smallest element are less than or equal to it, and all elements to the right
/// of the k-th smallest element are greater than or equal to it.
fn quick_partition<T: Number, C: Cluster<T>>(items: &mut [(T, &C)], k: usize) -> usize {
    qps(items, k, 0, items.len() - 1)
}

/// The recursive helper function for the Quick Partition algorithm.
fn qps<T: Number, C: Cluster<T>>(items: &mut [(T, &C)], k: usize, l: usize, r: usize) -> usize {
    if l >= r {
        min(l, r)
    } else {
        // Choose the pivot point
        let pivot = l + (r - l) / 2;
        let p = find_pivot(items, l, r, pivot);

        // Calculate the cumulative guaranteed cardinalities for the first p
        // `Cluster`s
        let cumulative_guarantees = items
            .iter()
            .take(p)
            .scan(0, |acc, (_, c)| {
                *acc += c.cardinality();
                Some(*acc)
            })
            .collect::<Vec<_>>();

        // Calculate the guaranteed cardinality of the p-th `Cluster`
        let guaranteed_p = if p > 0 { cumulative_guarantees[p - 1] } else { 0 };

        match guaranteed_p.cmp(&k) {
            Ordering::Equal => p,                      // Found the k-th smallest element
            Ordering::Less => qps(items, k, p + 1, r), // Need to look to the right
            Ordering::Greater => {
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

/// Moves pivot point and swaps elements around so that all elements to left
/// of pivot are less than or equal to pivot and all elements to right of pivot
/// are greater than pivot.
fn find_pivot<T, C>(items: &mut [(T, &C)], l: usize, r: usize, pivot: usize) -> usize
where
    T: Number,
    C: Cluster<T>,
{
    // Move pivot to the end
    items.swap(pivot, r);

    // Partition around pivot
    let (mut a, mut b) = (l, l);
    // Invariant: a <= b <= r
    while b < r {
        // If the current element is less than the pivot, swap it with the
        // element at a and increment a.
        if items[b].0 < items[r].0 {
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
