//! Ranged Nearest Neighbors search using a tree, as described in the CHESS
//! paper.

use distances::Number;
use rayon::prelude::*;

use crate::{
    cakes::{ParSearchable, Searchable},
    cluster::ParCluster,
    metric::ParMetric,
    Cluster, Metric,
};

use super::{ParSearchAlgorithm, RnnLinear, SearchAlgorithm};

/// Ranged Nearest Neighbors search using a tree.
pub struct RnnClustered<T: Number>(pub T);

impl<I, T: Number, C: Cluster<T>, M: Metric<I, T>, D: Searchable<I, T, C, M>> SearchAlgorithm<I, T, C, M, D>
    for RnnClustered<T>
{
    fn name(&self) -> &str {
        "RnnClustered"
    }

    fn radius(&self) -> Option<T> {
        Some(self.0)
    }

    fn k(&self) -> Option<usize> {
        None
    }

    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        let [confirmed, straddlers] = tree_search(data, metric, root, query, self.0);
        leaf_search(data, metric, confirmed, straddlers, query, self.0)
    }
}

impl<I: Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>, D: ParSearchable<I, T, C, M>>
    ParSearchAlgorithm<I, T, C, M, D> for RnnClustered<T>
{
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        let [confirmed, straddlers] = par_tree_search(data, metric, root, query, self.0);
        par_leaf_search(data, metric, confirmed, straddlers, query, self.0)
    }
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
#[inline(never)]
pub fn tree_search<'a, I, T, C, M, D>(data: &D, metric: &M, root: &'a C, query: &I, radius: T) -> [Vec<(&'a C, T)>; 2]
where
    T: Number + 'a,
    C: Cluster<T>,
    M: Metric<I, T>,
    D: Searchable<I, T, C, M>,
{
    let (overlap, distances) = root.overlaps_with(data, metric, query, radius);
    if !overlap {
        return [Vec::new(), Vec::new()];
    }

    let mut confirmed = Vec::new();
    let mut straddlers = Vec::new();
    let mut candidates = vec![(root, distances[0])];

    while !candidates.is_empty() {
        candidates = candidates.into_iter().fold(Vec::new(), |mut next_candidates, (c, d)| {
            if (c.radius() + d) <= radius {
                confirmed.push((c, d));
            } else if c.is_leaf() {
                straddlers.push((c, d));
            } else {
                next_candidates.extend(
                    c.overlapping_children(data, metric, query, radius)
                        .into_iter()
                        .map(|(c, ds)| (c, ds[0])),
                );
            }
            next_candidates
        });
    }

    [confirmed, straddlers]
}

/// Parallel version of [`tree_search`](crate::cakes::search::rnn_clustered::tree_search).
pub fn par_tree_search<'a, I, T, C, M, D>(
    data: &D,
    metric: &M,
    root: &'a C,
    query: &I,
    radius: T,
) -> [Vec<(&'a C, T)>; 2]
where
    I: Send + Sync,
    T: Number + 'a,
    C: ParCluster<T>,
    M: ParMetric<I, T>,
    D: ParSearchable<I, T, C, M>,
{
    let (overlap, distances) = root.par_overlaps_with(data, metric, query, radius);
    if !overlap {
        return [Vec::new(), Vec::new()];
    }

    let mut confirmed = Vec::new();
    let mut straddlers = Vec::new();
    let mut candidates = vec![(root, distances[0])];

    while !candidates.is_empty() {
        candidates = candidates.into_iter().fold(Vec::new(), |mut next_candidates, (c, d)| {
            if (c.radius() + d) <= radius {
                confirmed.push((c, d));
            } else if c.is_leaf() {
                straddlers.push((c, d));
            } else {
                next_candidates.extend(
                    c.par_overlapping_children(data, metric, query, radius)
                        .into_iter()
                        .map(|(c, ds)| (c, ds[0])),
                );
            }
            next_candidates
        });
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
#[inline(never)]
pub fn leaf_search<I, T, D, M, C>(
    data: &D,
    metric: &M,
    confirmed: Vec<(&C, T)>,
    straddlers: Vec<(&C, T)>,
    query: &I,
    radius: T,
) -> Vec<(usize, T)>
where
    T: Number,
    C: Cluster<T>,
    M: Metric<I, T>,
    D: Searchable<I, T, C, M>,
{
    confirmed
        .into_iter()
        .flat_map(|(c, d)| {
            if c.is_singleton() {
                c.indices().into_iter().map(|i| (i, d)).collect::<Vec<_>>()
            } else {
                data.query_to_all(metric, query, c).collect()
            }
        })
        .chain(
            straddlers
                .into_iter()
                .flat_map(|(c, _)| RnnLinear(radius).search(data, metric, c, query)),
        )
        .collect()
}

/// Parallel version of [`leaf_search`](crate::cakes::search::rnn_clustered::leaf_search).
pub fn par_leaf_search<I, T, C, M, D>(
    data: &D,
    metric: &M,
    confirmed: Vec<(&C, T)>,
    straddlers: Vec<(&C, T)>,
    query: &I,
    radius: T,
) -> Vec<(usize, T)>
where
    I: Send + Sync,
    T: Number,
    C: ParCluster<T>,
    M: ParMetric<I, T>,
    D: ParSearchable<I, T, C, M>,
{
    confirmed
        .into_par_iter()
        .flat_map(|(c, d)| {
            if c.is_singleton() {
                c.indices().into_iter().map(|i| (i, d)).collect::<Vec<_>>()
            } else {
                data.par_query_to_all(metric, query, c).collect()
            }
        })
        .chain(
            straddlers
                .into_par_iter()
                .flat_map(|(c, _)| RnnLinear(radius).par_search(data, metric, c, query)),
        )
        .collect()
}
