//! K-Nearest Neighbors search using a Depth First sieve.

use core::cmp::Reverse;

use distances::Number;

use crate::new_core::{Cluster, Dataset, ParDataset, SizedHeap};

/// K-Nearest Neighbors search using a Depth First sieve.
pub fn search<I, U, D, C>(data: &D, root: &C, query: &I, k: usize) -> Vec<(usize, U)>
where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    let mut candidates = SizedHeap::<(Reverse<U>, &C)>::new(None);
    let mut hits = SizedHeap::<(U, usize)>::new(Some(k));

    let d = root.distance_to_instance(data, query);
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
        pop_till_leaf(data, query, &mut candidates);
        leaf_into_hits(data, query, &mut hits, &mut candidates);
    }
    hits.items().map(|(d, i)| (i, d)).collect()
}

/// Parallel K-Nearest Neighbors search using a Depth First sieve.
pub fn par_search<I, U, D, C>(data: &D, root: &C, query: &I, k: usize) -> Vec<(usize, U)>
where
    I: Send + Sync,
    U: Number,
    D: ParDataset<I, U>,
    C: Cluster<U> + Send + Sync,
{
    let mut candidates = SizedHeap::<(Reverse<U>, &C)>::new(None);
    let mut hits = SizedHeap::<(U, usize)>::new(Some(k));

    let d = root.distance_to_instance(data, query);
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
pub fn d_min<U: Number, C: Cluster<U>>(c: &C, d: U) -> U {
    if d < c.radius() {
        U::ZERO
    } else {
        d - c.radius()
    }
}

/// Pops from the top of `candidates` until the top candidate is a leaf cluster.
fn pop_till_leaf<'a, I, U, D, C>(data: &D, query: &I, candidates: &mut SizedHeap<(Reverse<U>, &'a C)>)
where
    U: Number + 'a,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    while candidates
        .peek() // The top candidate is a leaf
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| !c.is_leaf())
    {
        let c = candidates
            .pop()
            .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| c);
        let children = c
            .children()
            .unwrap_or_else(|| unreachable!("elements are non-leaves"))
            .clusters();
        for c in children {
            let d = c.distance_to_instance(data, query);
            candidates.push((Reverse(d_min(c, d)), c));
        }
    }
}

/// Pops from the top of `candidates` and adds its points to `hits`.
fn leaf_into_hits<I, U, D, C>(
    data: &D,
    query: &I,
    hits: &mut SizedHeap<(U, usize)>,
    candidates: &mut SizedHeap<(Reverse<U>, &C)>,
) where
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    let (d, leaf) = candidates
        .pop()
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(Reverse(d), c)| (d, c));
    let indices = leaf.indices_post_permutation();
    if leaf.is_singleton() {
        indices.zip(std::iter::repeat(d)).for_each(|(i, d)| hits.push((d, i)));
    } else {
        indices
            .into_iter()
            .map(|i| (i, data.query_to_one(query, i)))
            .for_each(|(i, d)| hits.push((d, i)));
    };
}

/// Parallel version of `pop_till_leaf`.
fn par_pop_till_leaf<'a, I, U, D, C>(data: &D, query: &I, candidates: &mut SizedHeap<(Reverse<U>, &'a C)>)
where
    I: Send + Sync,
    U: Number + 'a,
    D: ParDataset<I, U>,
    C: Cluster<U> + Send + Sync,
{
    while candidates
        .peek() // The top candidate
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| !c.is_leaf())
    // is not a leaf
    {
        let c = candidates
            .pop()
            .map_or_else(|| unreachable!("`candidates` is non-empty"), |(_, c)| c);
        let children = c
            .children()
            .unwrap_or_else(|| unreachable!("elements are non-leaves"))
            .clusters();
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
    C: Cluster<U> + Send + Sync,
{
    let (d, leaf) = candidates
        .pop()
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(Reverse(d), c)| (d, c));
    if leaf.is_singleton() {
        leaf.indices_post_permutation()
            .zip(std::iter::repeat(d))
            .for_each(|(i, d)| hits.push((d, i)));
    } else {
        let indices = leaf.indices_post_permutation().collect::<Vec<_>>();
        data.par_query_to_many(query, &indices)
            .into_iter()
            .for_each(|(i, d)| hits.push((d, i)));
    };
}

#[cfg(test)]
mod tests {
    use crate::{
        cakes::search::tests::gen_grid_data,
        new_core::{
            cluster::{Ball, Partition},
            LinearSearch,
        },
    };

    use super::super::tests::{check_search_by_distance, gen_line_data};

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
            assert!(check_search_by_distance(true_hits.clone(), pred_hits, "KnnDepthFirst"));

            let pred_hits = par_search(&data, &root, query, k);
            assert_eq!(pred_hits.len(), k, "Parallel KNN search failed: {pred_hits:?}");
            assert!(check_search_by_distance(true_hits, pred_hits, "ParKnnDepthFirst"));
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

        for k in [1, 4, 8, 16, 32] {
            let true_hits = data.knn(query, k);
            assert_eq!(true_hits.len(), k, "Linear search failed: {true_hits:?}");

            let pred_hits = search(&data, &root, query, k);
            assert_eq!(pred_hits.len(), k, "KNN search failed: {pred_hits:?}");
            assert!(check_search_by_distance(
                true_hits.clone(),
                pred_hits,
                &format!("KnnDepthFirst k={k}")
            ));

            let pred_hits = par_search(&data, &root, query, k);
            assert_eq!(pred_hits.len(), k, "Parallel KNN search failed: {pred_hits:?}");
            assert!(check_search_by_distance(
                true_hits,
                pred_hits,
                &format!("KnnDepthFirst k={k}")
            ));
        }

        Ok(())
    }
}
