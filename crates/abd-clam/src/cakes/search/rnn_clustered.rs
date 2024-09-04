//! Ranged Nearest Neighbors search using the tree.

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

/// Clustered search for the ranged nearest neighbors of a query.
pub fn search<I, U, D, C>(data: &D, root: &C, query: &I, radius: U) -> Vec<(usize, U)>
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
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
    C: ParCluster<I, U, D>,
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
    C: Cluster<I, U, D>,
{
    let mut confirmed = Vec::new();
    let mut straddlers = Vec::new();
    let mut candidates = vec![root];

    let (mut terminal, mut non_terminal): (Vec<_>, Vec<_>);
    while !candidates.is_empty() {
        (terminal, non_terminal) = candidates
            .into_iter()
            .map(|c| (c, c.distance_to_center(data, query)))
            .filter(|&(c, d)| d <= (c.radius() + radius))
            .partition(|&(c, d)| (c.radius() + d) <= radius);
        confirmed.append(&mut terminal);

        (terminal, non_terminal) = non_terminal.into_iter().partition(|&(c, _)| c.is_leaf());
        straddlers.append(&mut terminal);

        candidates = non_terminal.into_iter().flat_map(|(c, _)| c.child_clusters()).collect();
    }

    [confirmed, straddlers]
}

/// Parallelized version of the tree search.
pub fn par_tree_search<'a, I, U, D, C>(data: &D, root: &'a C, query: &I, radius: U) -> [Vec<(&'a C, U)>; 2]
where
    I: Send + Sync,
    U: Number + 'a,
    D: ParDataset<I, U>,
    C: ParCluster<I, U, D>,
{
    let mut confirmed = Vec::new();
    let mut straddlers = Vec::new();
    let mut candidates = vec![root];

    let (mut terminal, mut non_terminal): (Vec<_>, Vec<_>);
    while !candidates.is_empty() {
        (terminal, non_terminal) = candidates
            .into_par_iter()
            .map(|c| (c, c.distance_to_center(data, query)))
            .filter(|&(c, d)| d <= (c.radius() + radius))
            .partition(|&(c, d)| (c.radius() + d) < radius);
        confirmed.append(&mut terminal);

        (terminal, non_terminal) = non_terminal.into_iter().partition(|&(c, _)| c.is_leaf());
        straddlers.append(&mut terminal);

        candidates = non_terminal.into_iter().flat_map(|(c, _)| c.child_clusters()).collect();
    }

    [confirmed, straddlers]
}

/// Perform fine-grained leaf search.
///
/// # Arguments
///
/// - `data` - The dataset to search.
/// - `confirmed` - The confirmed clusters from the tree search. All points
///   in these clusters are guaranteed to be within the query ball.
/// - `straddlers` - The straddlers from the tree search. These clusters
///   overlap the query ball, but not all points in the cluster are guaranteed
///   to be within the query ball.
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
    C: Cluster<I, U, D>,
{
    let hits = confirmed.into_iter().flat_map(|(c, d)| {
        if c.is_singleton() {
            c.indices().map(|i| (i, d)).collect()
        } else {
            c.distances_to_query(data, query)
        }
    });

    let distances = straddlers
        .into_iter()
        .flat_map(|(c, _)| c.distances_to_query(data, query))
        .filter(|&(_, d)| d <= radius);

    hits.chain(distances).collect()
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
    C: ParCluster<I, U, D>,
{
    let hits = confirmed.into_par_iter().flat_map(|(c, d)| {
        if c.is_singleton() {
            c.indices().map(|i| (i, d)).collect()
        } else {
            c.par_distances_to_query(data, query)
        }
    });

    let distances = straddlers
        .into_par_iter()
        .flat_map(|(c, _)| c.par_distances_to_query(data, query))
        .filter(|&(_, d)| d <= radius);

    hits.chain(distances).collect()
}

#[cfg(test)]
mod tests {
    use core::fmt::Debug;

    use distances::Number;

    use crate::Dataset;
    use crate::{
        adapter::BallAdapter, cakes::OffBall, cluster::ParCluster, partition::ParPartition, Ball, Cluster, FlatVec,
        Partition,
    };

    use crate::cakes::tests::{check_search_by_index, gen_grid_data, gen_line_data};

    pub fn check_rnn<I: Debug + Send + Sync, U: Number, C: ParCluster<I, U, FlatVec<I, U, usize>>>(
        root: &C,
        data: &FlatVec<I, U, usize>,
        query: &I,
        radius: U,
    ) -> bool {
        let true_hits = data.rnn(query, radius);

        let pred_hits = super::search(data, root, query, radius);
        assert_eq!(pred_hits.len(), true_hits.len(), "Rnn search failed: {pred_hits:?}");
        check_search_by_index(true_hits.clone(), pred_hits, "RnnClustered", data);

        let pred_hits = super::par_search(data, root, query, radius);
        assert_eq!(
            pred_hits.len(),
            true_hits.len(),
            "Parallel Rnn search failed: {pred_hits:?}"
        );
        check_search_by_index(true_hits, pred_hits, "Par RnnClustered", data);

        true
    }

    #[test]
    fn line() -> Result<(), String> {
        let data = gen_line_data(10)?;
        let query = &0;

        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let seed = Some(42);

        let ball = Ball::new_tree(&data, &criteria, seed);
        for radius in 0..=4 {
            assert!(check_rnn(&ball, &data, &query, radius));
        }

        let (off_ball, perm_data) = OffBall::from_ball_tree(ball, data);
        for radius in 0..=4 {
            assert!(check_rnn(&off_ball, &perm_data, &query, radius));
        }

        Ok(())
    }

    #[test]
    fn grid() -> Result<(), String> {
        let data = gen_grid_data(10)?;
        let query = &(0.0, 0.0);

        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let seed = Some(42);

        let ball = Ball::par_new_tree(&data, &criteria, seed);
        for radius in [1.0, 4.0, 8.0, 16.0, 32.0] {
            assert!(check_rnn(&ball, &data, &query, radius));
        }

        let (off_ball, perm_data) = OffBall::from_ball_tree(ball, data);
        for radius in [1.0, 4.0, 8.0, 16.0, 32.0] {
            assert!(check_rnn(&off_ball, &perm_data, &query, radius));
        }

        Ok(())
    }
}
