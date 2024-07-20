//! Ranged Nearest Neighbors search using the tree.

use distances::Number;
use rayon::prelude::*;

use crate::new_core::{Cluster, Dataset, ParDataset};

/// Clustered search for the ranged nearest neighbors of a query.
pub fn search<I, U, D, C>(data: &D, root: &C, query: &I, radius: U) -> Vec<(usize, U)>
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    let [confirmed, straddlers] = tree_search(data, root, query, radius);
    leaf_search(data, confirmed, straddlers, query, radius)
}

/// Parallel clustered search for the ranged nearest neighbors of a query.
pub fn par_search<I, U, D, C>(data: &D, root: &C, query: &I, radius: U) -> Vec<(usize, U)>
where
    I: Send + Sync,
    U: Number,
    D: ParDataset<I, U>,
    C: Cluster<U> + Send + Sync,
{
    let [confirmed, straddlers] = par_tree_search(data, root, query, radius);
    par_leaf_search(data, confirmed, straddlers, query, radius)
}

/// Perform coarse-grained tree search.
///
/// # Arguments
///
/// - `data` - The dataset to search.
/// - `root` - The root of the tree to search.
/// - `query` - The query to search around.
/// - `radius` - The radius to search within.
///
/// # Returns
///
/// A 2-slice of vectors of 2-tuples, where the first element in the slice
/// is the confirmed clusters, i.e. those that are contained within the
/// query ball, and the second element is the straddlers, i.e. those that
/// overlap the query ball. The 2-tuples are the clusters and the distance
/// from the query to the cluster center.
pub fn tree_search<'a, I, U, D, C>(data: &D, root: &'a C, query: &I, radius: U) -> [Vec<(&'a C, U)>; 2]
where
    U: Number + 'a,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    let mut confirmed = Vec::new();
    let mut straddlers = Vec::new();
    let mut candidates = vec![root];

    let (mut terminal, mut non_terminal): (Vec<_>, Vec<_>);
    while !candidates.is_empty() {
        (terminal, non_terminal) = candidates
            .into_iter()
            .map(|c| (c, c.distance_to_instance(data, query)))
            .filter(|&(c, d)| d <= (c.radius() + radius))
            .partition(|&(c, d)| (c.radius() + d) < radius);
        confirmed.append(&mut terminal);

        (terminal, non_terminal) = non_terminal.into_iter().partition(|&(c, _)| c.is_leaf());
        straddlers.append(&mut terminal);

        candidates = non_terminal
            .into_iter()
            .flat_map(|(c, _)| {
                c.children()
                    .unwrap_or_else(|| unreachable!("Non-leaf cluster without children"))
                    .clusters()
            })
            .collect();
    }

    [confirmed, straddlers]
}

/// Parallelized version of the tree search.
pub fn par_tree_search<'a, I, U, D, C>(data: &D, root: &'a C, query: &I, radius: U) -> [Vec<(&'a C, U)>; 2]
where
    I: Send + Sync,
    U: Number + 'a,
    D: ParDataset<I, U>,
    C: Cluster<U> + Send + Sync,
{
    let mut confirmed = Vec::new();
    let mut straddlers = Vec::new();
    let mut candidates = vec![root];

    let (mut terminal, mut non_terminal): (Vec<_>, Vec<_>);
    while !candidates.is_empty() {
        (terminal, non_terminal) = candidates
            .into_par_iter()
            .map(|c| (c, c.distance_to_instance(data, query)))
            .filter(|&(c, d)| d <= (c.radius() + radius))
            .partition(|&(c, d)| (c.radius() + d) < radius);
        confirmed.append(&mut terminal);

        (terminal, non_terminal) = non_terminal.into_par_iter().partition(|&(c, _)| c.is_leaf());
        straddlers.append(&mut terminal);

        candidates = non_terminal
            .into_par_iter()
            .flat_map(|(c, _)| {
                c.children()
                    .unwrap_or_else(|| unreachable!("Non-leaf cluster without children"))
                    .clusters()
            })
            .collect();
    }

    [confirmed, straddlers]
}

/// Perform fine-grained leaf search.
///
/// # Arguments
///
/// - `data` - The dataset to search.
/// - `confirmed` - The confirmed clusters from the tree search. All points
///  in these clusters are guaranteed to be within the query ball.
/// - `straddlers` - The straddlers from the tree search. These clusters
///  overlap the query ball, but not all points in the cluster are guaranteed
///  to be within the query ball.
/// - `query` - The query to search around.
/// - `radius` - The radius to search within.
///
/// # Returns
///
/// The `(index, distance)` pairs of the points within the query ball.
pub fn leaf_search<I, U, D, C>(
    data: &D,
    confirmed: Vec<(&C, U)>,
    straddlers: Vec<(&C, U)>,
    query: &I,
    radius: U,
) -> Vec<(usize, U)>
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    let hits = confirmed.into_iter().flat_map(|(c, d)| {
        if c.is_singleton() {
            c.indices_post_permutation().zip(core::iter::repeat(d)).collect()
        } else {
            data.query_to_many(query, &c.indices_post_permutation().collect::<Vec<_>>())
        }
    });

    let indices = straddlers
        .into_iter()
        .flat_map(|(c, _)| c.indices_post_permutation())
        .collect::<Vec<_>>();

    let distances = data.query_to_many(query, &indices);

    hits.chain(distances.into_iter().filter(|&(_, d)| d <= radius))
        .collect()
}

/// Parallelized version of the leaf search.
pub fn par_leaf_search<I, U, D, C>(
    data: &D,
    confirmed: Vec<(&C, U)>,
    straddlers: Vec<(&C, U)>,
    query: &I,
    radius: U,
) -> Vec<(usize, U)>
where
    I: Send + Sync,
    U: Number,
    D: ParDataset<I, U>,
    C: Cluster<U> + Send + Sync,
{
    let mut hits = confirmed
        .into_par_iter()
        .flat_map(|(c, d)| {
            if c.is_singleton() {
                c.indices_post_permutation().zip(core::iter::repeat(d)).collect()
            } else {
                data.par_query_to_many(query, &c.indices_post_permutation().collect::<Vec<_>>())
            }
        })
        .collect::<Vec<_>>();

    let indices = straddlers
        .into_par_iter()
        .flat_map(|(c, _)| c.indices_post_permutation())
        .collect::<Vec<_>>();

    let distances = data.par_query_to_many(query, &indices);

    hits.extend(distances.into_iter().filter(|&(_, d)| d <= radius));
    hits
}

#[cfg(test)]
mod tests {
    use crate::new_core::{
        cluster::{Ball, Partition},
        LinearSearch,
    };

    use super::super::tests::{check_search_by_index, gen_grid_data, gen_line_data};

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

        for radius in 0..=4 {
            let k = (2 * radius + 1) as usize;

            let true_hits = data.rnn(query, radius);
            assert_eq!(true_hits.len(), k, "Linear search failed: {true_hits:?}");

            let pred_hits = search(&data, &root, query, radius);
            assert_eq!(pred_hits.len(), k, "Clustered search failed: {pred_hits:?}");
            assert!(check_search_by_index(
                true_hits.clone(),
                pred_hits,
                &format!("RnnClustered {radius:.6}")
            ));

            let pred_hits = par_search(&data, &root, query, radius);
            assert_eq!(pred_hits.len(), k, "Parallel clustered search failed: {pred_hits:?}");
            assert!(check_search_by_index(
                true_hits,
                pred_hits,
                &format!("Par RnnClustered {radius:.6}")
            ));
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

        for radius in [1.0, 4.0, 8.0, 16.0, 32.0] {
            let true_hits = data.rnn(query, radius);

            let pred_hits = search(&data, &root, query, radius);
            assert_eq!(pred_hits.len(), true_hits.len(), "Rnn search failed: {pred_hits:?}");
            assert!(check_search_by_index(
                true_hits.clone(),
                pred_hits,
                &format!("RnnClustered {radius:.6}")
            ));

            let pred_hits = par_search(&data, &root, query, radius);
            assert_eq!(
                pred_hits.len(),
                true_hits.len(),
                "Parallel Rnn search failed: {pred_hits:?}"
            );
            assert!(check_search_by_index(
                true_hits,
                pred_hits,
                &format!("Par RnnClustered {radius:.6}")
            ));
        }

        Ok(())
    }
}
