//! K-Nearest Neighbors search using a Breadth First sieve.

use core::cmp::{min, Ordering};

use distances::Number;
use rayon::prelude::*;

use crate::{
    cluster::ParCluster,
    dataset::{ParDataset, SizedHeap},
    Cluster, Dataset,
};

/// K-Nearest Neighbors search using a Breadth First sieve.
pub fn search<I, U, D, C>(data: &D, root: &C, query: &I, k: usize) -> Vec<(usize, U)>
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
    let mut candidates = Vec::new();
    let mut hits = SizedHeap::<(U, usize)>::new(Some(k));

    let d = root.distance_to_center(data, query);
    candidates.push((d_max(root, d), root));

    while !candidates.is_empty() {
        let [needed, maybe_needed, _] = split_candidates(&mut candidates, k);

        let (leaves, parents) = needed
            .into_iter()
            .chain(maybe_needed)
            .partition::<Vec<_>, _>(|(_, c)| c.is_leaf());

        for (d, c) in leaves {
            if c.is_singleton() {
                c.indices().for_each(|i| hits.push((d, i)));
            } else {
                c.distances_to_query(data, query)
                    .into_iter()
                    .for_each(|(i, d)| hits.push((d, i)));
            }
        }

        candidates = Vec::new();
        for (_, p) in parents {
            p.child_clusters()
                .map(|c| (c, c.distance_to_center(data, query)))
                .for_each(|(c, d)| candidates.push((d_max(c, d), c)));
        }
    }

    hits.items().map(|(d, i)| (i, d)).collect()
}

/// Parallel K-Nearest Neighbors search using a Breadth First sieve.
pub fn par_search<I, U, D, C>(data: &D, root: &C, query: &I, k: usize) -> Vec<(usize, U)>
where
    I: Send + Sync,
    U: Number,
    D: ParDataset<I, U>,
    C: ParCluster<I, U, D>,
{
    let mut candidates = Vec::new();
    let mut hits = SizedHeap::<(U, usize)>::new(Some(k));

    let d = root.distance_to_center(data, query);
    candidates.push((d_max(root, d), root));

    while !candidates.is_empty() {
        let [needed, maybe_needed, _] = split_candidates(&mut candidates, k);

        let (leaves, parents) = needed
            .into_iter()
            .chain(maybe_needed)
            .partition::<Vec<_>, _>(|(_, c)| c.is_leaf());

        for (d, c) in leaves {
            if c.is_singleton() {
                c.indices().for_each(|i| hits.push((d, i)));
            } else {
                c.par_distances_to_query(data, query)
                    .into_iter()
                    .for_each(|(i, d)| hits.push((d, i)));
            }
        }

        candidates = Vec::new();
        let distances = parents
            .into_par_iter()
            .flat_map(|(_, p)| p.child_clusters().collect::<Vec<_>>())
            .map(|c| (c, c.distance_to_center(data, query)))
            .collect::<Vec<_>>();
        distances
            .into_iter()
            .for_each(|(c, d)| candidates.push((d_max(c, d), c)));
    }

    hits.items().map(|(d, i)| (i, d)).collect()
}

/// Returns the theoretical maximum distance from the query to a point in the cluster.
fn d_max<I, U: Number, D: Dataset<I, U>, C: Cluster<I, U, D>>(c: &C, d: U) -> U {
    c.radius() + d
}

/// Splits the candidates three ways: those needed to get to k hits, those that
/// might be needed to get to k hits, and those that are not needed to get to k
/// hits.
fn split_candidates<'a, I, U, D, C>(candidates: &mut [(U, &'a C)], k: usize) -> [Vec<(U, &'a C)>; 3]
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
    let threshold_index = quick_partition(candidates, k);
    let threshold = candidates[threshold_index].0;

    let (needed, others) = candidates.iter().partition::<Vec<_>, _>(|(d, _)| *d < threshold);

    let (not_needed, maybe_needed) = others
        .into_iter()
        .map(|(d, c)| {
            let diam = c.radius().double();
            if d <= diam {
                (d, U::ZERO, c)
            } else {
                (d, d - diam, c)
            }
        })
        .partition::<Vec<_>, _>(|(_, d, _)| *d > threshold);

    let not_needed = not_needed.into_iter().map(|(d, _, c)| (d, c)).collect();
    let maybe_needed = maybe_needed.into_iter().map(|(d, _, c)| (d, c)).collect();

    [needed, maybe_needed, not_needed]
}

/// The Quick Partition algorithm, which is a variant of the Quick Select
/// algorithm. It finds the k-th smallest element in a list of elements, while
/// also reordering the list so that all elements to the left of the k-th
/// smallest element are less than or equal to it, and all elements to the right
/// of the k-th smallest element are greater than or equal to it.
fn quick_partition<I, U, C, D>(items: &mut [(U, &C)], k: usize) -> usize
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
    qps(items, k, 0, items.len() - 1)
}

/// The recursive helper function for the Quick Partition algorithm.
fn qps<I, U, D, C>(items: &mut [(U, &C)], k: usize, l: usize, r: usize) -> usize
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
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
fn find_pivot<I, U, C, D>(items: &mut [(U, &C)], l: usize, r: usize, pivot: usize) -> usize
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
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

#[cfg(test)]
mod tests {
    use crate::{
        adapter::BallAdapter,
        cakes::OffBall,
        cluster::{Ball, Partition},
        Cluster,
    };

    use super::super::knn_depth_first::tests::check_knn;
    use crate::cakes::tests::{gen_grid_data, gen_line_data};

    #[test]
    fn line() -> Result<(), String> {
        let data = gen_line_data(10)?;
        let query = &0;

        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let seed = Some(42);

        let ball = Ball::new_tree(&data, &criteria, seed);
        for k in [1, 4, 8] {
            assert!(check_knn(&ball, &data, query, k));
        }

        let (off_ball, perm_data) = OffBall::from_ball_tree(ball, data, true);
        for k in [1, 4, 8] {
            assert!(check_knn(&off_ball, &perm_data, query, k));
        }

        Ok(())
    }

    #[test]
    fn grid() -> Result<(), String> {
        let data = gen_grid_data(10)?;
        let query = &(0.0, 0.0);

        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let seed = Some(42);

        let ball = Ball::new_tree(&data, &criteria, seed);
        for k in [1, 4, 8] {
            assert!(check_knn(&ball, &data, query, k));
        }

        let (off_ball, perm_data) = OffBall::from_ball_tree(ball, data, true);
        for k in [1, 4, 8] {
            assert!(check_knn(&off_ball, &perm_data, query, k));
        }

        Ok(())
    }
}
