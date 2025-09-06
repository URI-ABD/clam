//! K-Nearest Neighbors search using a Depth First sieve.

use core::cmp::Reverse;

use rayon::prelude::*;

use crate::{Cluster, Dataset, DistanceValue, ParCluster, ParDataset, SizedHeap};

use super::{ParSearchAlgorithm, SearchAlgorithm};

/// K-Nearest Neighbors search using a Depth First sieve.
pub struct KnnDepthFirst(pub usize);

impl<I, T: DistanceValue, C: Cluster<T>, M: Fn(&I, &I) -> T, D: Dataset<I>> SearchAlgorithm<I, T, C, M, D>
    for KnnDepthFirst
{
    fn name(&self) -> &'static str {
        "KnnDepthFirst"
    }

    fn radius(&self) -> Option<T> {
        None
    }

    fn k(&self) -> Option<usize> {
        Some(self.0)
    }

    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        let mut candidates = SizedHeap::<(Reverse<T>, &C)>::new(None);
        let mut hits = SizedHeap::<(T, usize)>::new(Some(self.0));

        let d = data.query_to_one(query, root.arg_center(), metric);
        candidates.push((Reverse(d_min(root, d)), root));

        while !hits.is_full()  // We do not have enough hits.
            || (!candidates.is_empty()  // We have candidates.
                && hits  // and
                    .peek()
                    .map_or_else(|| unreachable!("`hits` is non-empty."), |(d, _)| *d)  // the farthest hit
                    >= candidates  // is farther than
                        .peek() // the closest candidate
                        .map_or_else(|| unreachable!("`candidates` is non-empty."), |(d, _)| d.0))
        {
            let (d, leaf) = pop_till_leaf(data, metric, query, &mut candidates);
            leaf_into_hits(data, metric, query, &mut hits, d, leaf);
        }
        hits.items().map(|(d, i)| (i, d)).collect()
    }
}

impl<
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        C: ParCluster<T>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        D: ParDataset<I>,
    > ParSearchAlgorithm<I, T, C, M, D> for KnnDepthFirst
{
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        let mut candidates = SizedHeap::<(Reverse<T>, &C)>::new(None);
        let mut hits = SizedHeap::<(T, usize)>::new(Some(self.0));

        let d = data.query_to_one(query, root.arg_center(), metric);
        candidates.push((Reverse(d_min(root, d)), root));

        while !hits.is_full()  // We do not have enough hits.
            || (!candidates.is_empty()  // We have candidates.
                && hits  // and
                    .peek()
                    .map_or_else(|| unreachable!("`hits` is non-empty."), |(d, _)| *d)  // the farthest hit
                    >= candidates  // is farther than
                        .peek() // the closest candidate
                        .map_or_else(|| unreachable!("`candidates` is non-empty."), |(d, _)| d.0))
        {
            par_pop_till_leaf(data, metric, query, &mut candidates);
            par_leaf_into_hits(data, metric, query, &mut hits, &mut candidates);
        }
        hits.items().map(|(d, i)| (i, d)).collect()
    }
}

/// Calculates the theoretical best case distance for a point in a cluster, i.e.,
/// the closest a point in a given cluster could possibly be to the query.
pub fn d_min<T: DistanceValue, C: Cluster<T>>(c: &C, d: T) -> T {
    if d < c.radius() {
        T::zero()
    } else {
        d - c.radius()
    }
}

/// Pops from the top of `candidates` until the top candidate is a leaf cluster.
/// Then, pops and returns the leaf cluster.
fn pop_till_leaf<'a, I, T, C, M, D>(
    data: &D,
    metric: &M,
    query: &I,
    candidates: &mut SizedHeap<(Reverse<T>, &'a C)>,
) -> (T, &'a C)
where
    T: DistanceValue + 'a,
    C: Cluster<T>,
    M: Fn(&I, &I) -> T,
    D: Dataset<I>,
{
    while candidates
        .peek() // The top candidate
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| !c.is_leaf())
    // is not a leaf
    {
        let parent = candidates
            .pop()
            .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| c);
        parent.children().into_iter().for_each(|child| {
            candidates.push((
                Reverse(d_min(child, data.query_to_one(query, child.arg_center(), metric))),
                child,
            ));
        });
    }
    candidates
        .pop()
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(Reverse(d), c)| (d, c))
}

/// Parallel version of [`pop_till_leaf`](crate::cakes::search::knn_depth_first::pop_till_leaf).
fn par_pop_till_leaf<'a, I, T, C, M, D>(data: &D, metric: &M, query: &I, candidates: &mut SizedHeap<(Reverse<T>, &'a C)>)
where
    I: Send + Sync,
    T: DistanceValue + Send + Sync + 'a,
    C: ParCluster<T>,
    M: (Fn(&I, &I) -> T) + Send + Sync,
    D: ParDataset<I>,
{
    while candidates
        .peek() // The top candidate
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| !c.is_leaf())
    // is not a leaf
    {
        let parent = candidates
            .pop()
            .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| c);
        parent
            .children()
            .into_par_iter()
            .map(|child| (child, data.query_to_one(query, child.arg_center(), metric)))
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|(child, d)| candidates.push((Reverse(d_min(child, d)), child)));
    }
}

/// Pops from the top of `candidates` and adds its points to `hits`.
fn leaf_into_hits<I, T, C, M, D>(data: &D, metric: &M, query: &I, hits: &mut SizedHeap<(T, usize)>, d: T, leaf: &C)
where
    T: DistanceValue,
    C: Cluster<T>,
    M: Fn(&I, &I) -> T,
    D: Dataset<I>,
{
    if leaf.is_singleton() {
        leaf.indices().into_iter().for_each(|i| hits.push((d, i)));
    } else {
        data.query_to_many(query, leaf.indices(), metric)
            .into_iter()
            .for_each(|(i, d)| hits.push((d, i)));
    }
}

/// Parallel version of [`leaf_into_hits`](crate::cakes::search::knn_depth_first::leaf_into_hits).
fn par_leaf_into_hits<I, T, C, M, D>(
    data: &D,
    metric: &M,
    query: &I,
    hits: &mut SizedHeap<(T, usize)>,
    candidates: &mut SizedHeap<(Reverse<T>, &C)>,
) where
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    C: ParCluster<T>,
    M: (Fn(&I, &I) -> T) + Send + Sync,
    D: ParDataset<I>,
{
    let (d, leaf) = candidates
        .pop()
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(Reverse(d), c)| (d, c));
    if leaf.is_singleton() {
        leaf.indices().into_iter().for_each(|i| hits.push((d, i)));
    } else {
        data.par_query_to_many(query, leaf.indices(), metric)
            .into_iter()
            .for_each(|(i, d)| hits.push((d, i)));
    }
}
