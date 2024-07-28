//! K-Nearest Neighbors search using a Breadth First sieve.

use core::cmp::Reverse;

use distances::Number;
use rayon::prelude::*;

use crate::core::{Cluster, Dataset, ParDataset, SizedHeap};

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
            p.children()
                .unwrap_or_else(|| unreachable!("This is only called on non-leaves."))
                .clusters()
                .into_iter()
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
    C: Cluster<U> + Send + Sync,
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
            .flat_map(|(_, p)| p.children().unwrap_or_else(|| unreachable!()).clusters())
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
    use crate::core::{
        cluster::{Ball, Partition},
        LinearSearch,
    };

    use super::super::tests::{check_search_by_distance, gen_grid_data, gen_line_data};

    use super::*;

    #[test]
    fn line() -> Result<(), String> {
        let mut data = gen_line_data(10)?;
        let query = &0;

        let criteria = |c: &Ball<u32>| c.cardinality() > 1;
        let seed = Some(42);
        let (root, indices) = Ball::new_tree_and_permute(&mut data, &criteria, seed);

        println!();
        println!("Re-joined Indices: {indices:?}");
        println!(
            "Instances: {:?}",
            (0..data.cardinality())
                .zip(data.instances.iter())
                .zip(data.metadata.iter())
                .map(|((i, x), m)| (i, x, m))
                .collect::<Vec<_>>()
        );

        for k in [1, 4, 8] {
            let true_hits = data.knn(query, k);
            assert_eq!(true_hits.len(), k, "Linear search failed: {true_hits:?}");

            let pred_hits = search(&data, &root, query, k);
            assert_eq!(pred_hits.len(), k, "KNN search failed: {pred_hits:?}");
            assert!(check_search_by_distance(
                true_hits.clone(),
                pred_hits,
                "KnnBreadthFirst"
            ));

            let pred_hits = par_search(&data, &root, query, k);
            assert_eq!(pred_hits.len(), k, "Parallel KNN search failed: {pred_hits:?}");
            assert!(check_search_by_distance(true_hits, pred_hits, "ParKnnBreadthFirst"));
        }

        Ok(())
    }

    #[test]
    fn grid() -> Result<(), String> {
        let mut data = gen_grid_data(10)?;
        let query = &(0.0, 0.0);

        let criteria = |c: &Ball<f32>| c.cardinality() > 1;
        let seed = Some(42);
        let (root, _) = Ball::new_tree_and_permute(&mut data, &criteria, seed);

        // for k in [1, 4, 8, 16, 32] {
        for k in [8] {
            let true_hits = data.knn(query, k);
            assert_eq!(true_hits.len(), k, "Linear search failed: {true_hits:?}");

            let pred_hits = search(&data, &root, query, k);
            assert_eq!(pred_hits.len(), k, "KNN search failed: {pred_hits:?}");
            assert!(check_search_by_distance(
                true_hits.clone(),
                pred_hits,
                &format!("KnnBreadthFirst k={k}")
            ));

            let pred_hits = par_search(&data, &root, query, k);
            assert_eq!(pred_hits.len(), k, "Parallel KNN search failed: {pred_hits:?}");
            assert!(check_search_by_distance(
                true_hits,
                pred_hits,
                &format!("KnnBreadthFirst k={k}")
            ));
        }

        Ok(())
    }
}
