//! K-Nearest Neighbors search using a Breadth First sieve.

use core::cmp::{min, Ordering, Reverse};

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, dataset::ParDataset, linear_search::SizedHeap, Cluster, Dataset};

/// K-Nearest Neighbors search using a Breadth First sieve.
pub fn search<I, U, D, C>(data: &D, root: &C, query: &I, k: usize) -> Vec<(usize, U)>
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
    let mut candidates = SizedHeap::<(Reverse<U>, &C)>::new(None);
    let mut hits = SizedHeap::<(U, usize)>::new(Some(k));

    let d = root.distance_to_center(data, query);
    candidates.push((Reverse(d_max(root, d)), root));

    while !candidates.is_empty() {
        let [needed, maybe_needed, _] = split_candidates(&hits, candidates);

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

        candidates = SizedHeap::new(None);
        for (_, p) in parents {
            p.child_clusters()
                .map(|c| (c, c.distance_to_center(data, query)))
                .for_each(|(c, d)| candidates.push((Reverse(d_max(c, d)), c)));
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
    let mut candidates = SizedHeap::<(Reverse<U>, &C)>::new(None);
    let mut hits = SizedHeap::<(U, usize)>::new(Some(k));

    let d = root.distance_to_center(data, query);
    candidates.push((Reverse(d_max(root, d)), root));

    while !candidates.is_empty() {
        let [needed, maybe_needed, _] = split_candidates(&hits, candidates);

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

        candidates = SizedHeap::new(None);
        let distances = parents
            .into_par_iter()
            .flat_map(|(_, p)| p.child_clusters().collect::<Vec<_>>())
            .map(|c| (c, c.distance_to_center(data, query)))
            .collect::<Vec<_>>();
        distances
            .into_iter()
            .for_each(|(c, d)| candidates.push((Reverse(d_max(c, d)), c)));
    }

    hits.items().map(|(d, i)| (i, d)).collect()
}

/// Returns the theoretical maximum distance from the query to a point in the cluster.
fn d_max<I, U: Number, D: Dataset<I, U>, C: Cluster<I, U, D>>(c: &C, d: U) -> U {
    c.radius() + d
}

/// Wrapper for `_partition_items`.
fn partition_items<I, U, C, D>(items: &mut Vec<(U, &C)>, k: usize) -> usize
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
    _partition_items(items, k, 0, items.len() - 1)
}

/// Finds the smallest index i such that the combined cardinality of items which
/// are at least as close to the query as items[i] is at least k.
fn _partition_items<I, U, D, C>(items: &mut Vec<(U, &C)>, k: usize, l: usize, r: usize) -> usize
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
    if l >= r {
        min(l, r)
    } else {
        let pivot = l + (r - l) / 2;
        let p = partition_once(items, l, r, pivot);

        let guaranteed_p = items.iter().take(p).map(|(_, c)| c.cardinality()).sum::<usize>();
        let guaranteed_p_minus_one = items.iter().take(p - 1).map(|(_, c)| c.cardinality()).sum::<usize>();
        match guaranteed_p.cmp(&k) {
            Ordering::Equal => p,
            Ordering::Less => _partition_items(items, k, p + 1, r),
            Ordering::Greater => {
                if p == 0 || guaranteed_p_minus_one < k {
                    p
                } else {
                    _partition_items(items, k, l, p - 1)
                }
            }
        }
    }
}

/// Changes pivot point and swaps elements around so that all elements to left
/// of pivot are less than or equal to pivot and all elements to right of pivot
/// are greater than pivot.
fn partition_once<I, U, C, D>(items: &mut [(U, &C)], l: usize, r: usize, pivot: usize) -> usize
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
    items.swap(pivot, r);

    let (mut a, mut b) = (l, l);
    while b < r {
        if items[b].0 < items[r].0 {
            items.swap(a, b);
            a += 1;
        }
        b += 1;
    }

    items.swap(a, r);

    a
}

/// Splits the candidates three ways: those needed to get to k hits, those that
/// might be needed to get to k hits, and those that are not needed to get to k
/// hits.
fn split_candidates<'a, I, U, D, C>(
    hits: &SizedHeap<(U, usize)>,
    candidates: SizedHeap<(Reverse<U>, &'a C)>,
) -> [Vec<(U, &'a C)>; 3]
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
    let k = hits
        .k()
        .unwrap_or_else(|| unreachable!("The `hits` heap should have a maximum size."));
    let mut items = candidates.items().map(|(Reverse(d), c)| (d, c)).collect::<Vec<_>>();

    let threshold_index = partition_items(&mut items, k);
    let threshold = items[threshold_index].0;

    let (needed, items) = items.into_iter().partition::<Vec<_>, _>(|(d, _)| *d < threshold);

    let (not_needed, maybe_needed) = items
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

        let (off_ball, perm_data) = OffBall::from_ball_tree(ball, data);
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

        let (off_ball, perm_data) = OffBall::from_ball_tree(ball, data);
        for k in [1, 4, 8] {
            assert!(check_knn(&off_ball, &perm_data, query, k));
        }

        Ok(())
    }
}
