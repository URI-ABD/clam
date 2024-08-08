//! K-Nearest Neighbors search using a Depth First sieve.

use core::cmp::Reverse;

use distances::Number;

use crate::{cluster::ParCluster, dataset::ParDataset, linear_search::SizedHeap, Cluster, Dataset};

/// K-Nearest Neighbors search using a Depth First sieve.
pub fn search<I, U, D, C>(data: &D, root: &C, query: &I, k: usize) -> Vec<(usize, U)>
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
    let mut candidates = SizedHeap::<(Reverse<U>, &C)>::new(None);
    let mut hits = SizedHeap::<(U, usize)>::new(Some(k));

    let d = root.distance_to_center(data, query);
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
        let (d, leaf) = pop_till_leaf(data, query, &mut candidates);
        leaf_into_hits(data, query, &mut hits, d, leaf);
    }
    hits.items().map(|(d, i)| (i, d)).collect()
}

/// Parallel K-Nearest Neighbors search using a Depth First sieve.
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
        par_pop_till_leaf(data, query, &mut candidates);
        par_leaf_into_hits(data, query, &mut hits, &mut candidates);
    }
    hits.items().map(|(d, i)| (i, d)).collect()
}

/// Calculates the theoretical best case distance for a point in a cluster, i.e.,
/// the closest a point in a given cluster could possibly be to the query.
pub fn d_min<I, U: Number, D: Dataset<I, U>, C: Cluster<I, U, D>>(c: &C, d: U) -> U {
    if d < c.radius() {
        U::ZERO
    } else {
        d - c.radius()
    }
}

/// Pops from the top of `candidates` until the top candidate is a leaf cluster.
/// Then, pops and returns the leaf cluster.
fn pop_till_leaf<'a, I, U, D, C>(data: &D, query: &I, candidates: &mut SizedHeap<(Reverse<U>, &'a C)>) -> (U, &'a C)
where
    U: Number + 'a,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
    while candidates
        .peek() // The top candidate is a leaf
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| !c.is_leaf())
    {
        let parent = candidates
            .pop()
            .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| c);
        for child in parent.child_clusters() {
            candidates.push((Reverse(d_min(child, child.distance_to_center(data, query))), child));
        }
    }
    candidates
        .pop()
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(Reverse(d), c)| (d, c))
}

/// Pops from the top of `candidates` and adds its points to `hits`.
fn leaf_into_hits<I, U, D, C>(data: &D, query: &I, hits: &mut SizedHeap<(U, usize)>, d: U, leaf: &C)
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<I, U, D>,
{
    if leaf.is_singleton() {
        leaf.repeat_distance(d).into_iter().for_each(|(i, d)| hits.push((d, i)));
    } else {
        leaf.distances(data, query)
            .into_iter()
            .for_each(|(i, d)| hits.push((d, i)));
    };
}

/// Parallel version of `pop_till_leaf`.
fn par_pop_till_leaf<'a, I, U, D, C>(data: &D, query: &I, candidates: &mut SizedHeap<(Reverse<U>, &'a C)>)
where
    I: Send + Sync,
    U: Number + 'a,
    D: ParDataset<I, U>,
    C: ParCluster<I, U, D>,
{
    while candidates
        .peek() // The top candidate
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| !c.is_leaf())
    // is not a leaf
    {
        let parent = candidates
            .pop()
            .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| c);
        let children = parent.child_clusters().collect::<Vec<_>>();
        let indices = children.iter().map(|c| c.arg_center()).collect::<Vec<_>>();
        data.par_query_to_many(query, &indices)
            .into_iter()
            .zip(children)
            .for_each(|((_, d), c)| candidates.push((Reverse(d_min(c, d)), c)));
    }
}

/// Parallel version of `leaf_into_hits`.
fn par_leaf_into_hits<I, U, D, C>(
    data: &D,
    query: &I,
    hits: &mut SizedHeap<(U, usize)>,
    candidates: &mut SizedHeap<(Reverse<U>, &C)>,
) where
    I: Send + Sync,
    U: Number,
    D: ParDataset<I, U>,
    C: ParCluster<I, U, D>,
{
    let (d, leaf) = candidates
        .pop()
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(Reverse(d), c)| (d, c));
    if leaf.is_singleton() {
        leaf.repeat_distance(d).into_iter().for_each(|(i, d)| hits.push((d, i)));
    } else {
        leaf.par_distances(data, query)
            .into_iter()
            .for_each(|(i, d)| hits.push((d, i)));
    };
}

#[cfg(test)]
pub(crate) mod tests {
    use distances::Number;

    use crate::{
        cakes::OffBall,
        cluster::{Ball, ParCluster, Partition},
        linear_search::LinearSearch,
        Cluster, FlatVec,
    };

    use super::super::tests::{check_search_by_distance, gen_grid_data, gen_line_data};

    pub fn check_knn<I: Send + Sync, U: Number, C: ParCluster<I, U, FlatVec<I, U, usize>>>(
        root: &C,
        data: &FlatVec<I, U, usize>,
        query: &I,
        k: usize,
    ) -> bool {
        let true_hits = data.knn(query, k);

        let pred_hits = super::search(data, root, query, k);
        assert_eq!(pred_hits.len(), true_hits.len(), "Knn search failed: {pred_hits:?}");
        check_search_by_distance(true_hits.clone(), pred_hits, "KnnClustered");

        let pred_hits = super::par_search(data, root, query, k);
        assert_eq!(
            pred_hits.len(),
            true_hits.len(),
            "Parallel Knn search failed: {pred_hits:?}"
        );
        check_search_by_distance(true_hits.clone(), pred_hits, "Par KnnClustered");

        true
    }

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

        let mut data = data;
        let root = OffBall::from_ball_tree(ball, &mut data);

        for k in [1, 4, 8] {
            assert!(check_knn(&root, &data, query, k));
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

        let mut data = data;
        let root = OffBall::from_ball_tree(ball, &mut data);

        for k in [1, 4, 8] {
            assert!(check_knn(&root, &data, query, k));
        }

        Ok(())
    }
}
