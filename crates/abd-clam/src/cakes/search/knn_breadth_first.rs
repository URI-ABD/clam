//! K-Nearest Neighbors search using a Breadth First sieve.

use core::cmp::Reverse;

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, dataset::ParDataset, linear_search::SizedHeap, Cluster, Dataset};

/// K-Nearest Neighbors search using a Breadth First sieve.
pub fn search<I, U, D, C>(data: &D, root: &C, query: &I, k: usize) -> Vec<(usize, U)>
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<U>,
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
                c.repeat_distance(d).into_iter().for_each(|(i, d)| hits.push((d, i)));
            } else {
                c.distances(data, query)
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
    C: ParCluster<U>,
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
                c.repeat_distance(d).into_iter().for_each(|(i, d)| hits.push((d, i)));
            } else {
                c.par_distances(data, query)
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
fn d_max<U: Number, C: Cluster<U>>(c: &C, d: U) -> U {
    c.radius() + d
}

/// Splits the candidates three ways: those needed to get to k hits, those that
/// might be needed to get to k hits, and those that are not needed to get to k
/// hits.
fn split_candidates<'a, U, C>(
    hits: &SizedHeap<(U, usize)>,
    candidates: SizedHeap<(Reverse<U>, &'a C)>,
) -> [Vec<(U, &'a C)>; 3]
where
    U: Number,
    C: Cluster<U>,
{
    let k = hits
        .k()
        .unwrap_or_else(|| unreachable!("The `hits` heap should have a maximum size."));
    let items = candidates.items().map(|(Reverse(d), c)| (d, c)).collect::<Vec<_>>();
    let threshold_index = items
        .iter()
        .scan(hits.len(), |num_hits_so_far, (_, c)| {
            *num_hits_so_far += c.cardinality();
            Some(*num_hits_so_far)
        })
        .position(|num_hits| num_hits > k)
        .unwrap_or_else(|| items.len() - 1);
    let threshold = {
        let (d, _) = items[threshold_index];
        let kth_distance = hits.peek().map_or(U::ZERO, |(d, _)| *d);
        if d < kth_distance {
            kth_distance
        } else {
            d
        }
    };

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
        cakes::{cluster::SquishyBall, OffBall},
        cluster::{Ball, Partition},
        Cluster,
    };

    use super::super::{
        knn_depth_first::tests::check_knn,
        tests::{gen_grid_data, gen_line_data},
    };

    #[test]
    fn line() -> Result<(), String> {
        let data = gen_line_data(10)?;
        let query = &0;

        let criteria = |c: &Ball<u32>| c.cardinality() > 1;
        let seed = Some(42);

        let ball = Ball::new_tree(&data, &criteria, seed);
        let squishy_ball = SquishyBall::from_root(ball.clone(), &data, true);

        for k in [1, 4, 8] {
            assert!(check_knn(&ball, &squishy_ball, &data, query, k));
        }

        let mut data = data;
        let root = OffBall::from_ball_tree(ball, &mut data);
        let squishy_root = SquishyBall::from_root(root.clone(), &data, true);

        for k in [1, 4, 8] {
            assert!(check_knn(&root, &squishy_root, &data, query, k));
        }

        Ok(())
    }

    #[test]
    fn grid() -> Result<(), String> {
        let data = gen_grid_data(10)?;
        let query = &(0.0, 0.0);

        let criteria = |c: &Ball<f32>| c.cardinality() > 1;
        let seed = Some(42);

        let ball = Ball::new_tree(&data, &criteria, seed);
        let squishy_ball = SquishyBall::from_root(ball.clone(), &data, true);

        for k in [1, 4, 8] {
            assert!(check_knn(&ball, &squishy_ball, &data, query, k));
        }

        let mut data = data;
        let root = OffBall::from_ball_tree(ball, &mut data);
        let squishy_root = SquishyBall::from_root(root.clone(), &data, true);

        for k in [1, 4, 8] {
            assert!(check_knn(&root, &squishy_root, &data, query, k));
        }

        Ok(())
    }
}
