//! Search function and helper functions for knn with expanding threshold.

use distances::Number;

use crate::{Cluster, Dataset, Instance, Tree};

use super::{OrdNumber, RevNumber};

/// K-Nearest Neighbor search with expanding threshold.
///
/// /// # Arguments
///
/// * `tree` - The tree to search.
/// * `query` - The query to search around.
/// * `k` - The number of neighbors to search for.
///
/// # Returns
///
/// A vector of 2-tuples, where the first element is the index of the instance
/// and the second element is the distance from the query to the instance.
///
/// Contrast this to `SieveV1` and `SieveV2`, which use a (mostly) decreasing threshold.
pub fn search<I, U, D, C>(tree: &Tree<I, U, D, C>, query: &I, k: usize) -> Vec<(usize, U)>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    let mut candidates = priority_queue::PriorityQueue::<&C, RevNumber<U>>::new();
    let mut hits = priority_queue::PriorityQueue::<usize, OrdNumber<U>>::new();

    let (data, root) = (tree.data(), &tree.root);

    let d = root.distance_to_instance(data, query);
    candidates.push(root, RevNumber(d_min(root, d)));

    // Stop if we have enough hits and the farthest hit is closer than the closest cluster (closeness determined by d_min).
    while hits.len() < k
        || (!candidates.is_empty()
            && hits
                .peek()
                .map_or_else(|| unreachable!("`hits` is non-empty."), |(_, &OrdNumber(d))| d)
                >= candidates
                    .peek()
                    .map_or_else(|| unreachable!("`candidates` is non-empty."), |(_, &RevNumber(d))| d))
    {
        pop_till_leaf(tree, query, &mut candidates);
        leaf_into_hits(tree, query, &mut hits, &mut candidates);
        trim_hits(k, &mut hits);
    }
    hits.into_iter().map(|(i, OrdNumber(d))| (i, d)).collect()
}

/// Calculates the theoretical best case distance for a point in a cluster, i.e.,
/// the closest a point in a given cluster could possibly be to the query.
pub fn d_min<U: Number, C: Cluster<U>>(c: &C, d: U) -> U {
    if d < c.radius() {
        U::zero()
    } else {
        d - c.radius()
    }
}

/// Pops from the top of `candidates` until the top candidate is a leaf cluster.
fn pop_till_leaf<I, U, D, C>(
    tree: &Tree<I, U, D, C>,
    query: &I,
    candidates: &mut priority_queue::PriorityQueue<&C, RevNumber<U>>,
) where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    while !candidates
        .peek()
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(c, _)| c.is_leaf())
    {
        let [l, r] = candidates.pop().map_or_else(
            || unreachable!("`candidates` is non-empty"),
            |(c, _)| c.children().unwrap_or_else(|| unreachable!("elements are non-leaves")),
        );
        let [dl, dr] = [
            l.distance_to_instance(tree.data(), query),
            r.distance_to_instance(tree.data(), query),
        ];
        candidates.push(l, RevNumber(d_min(l, dl)));
        candidates.push(r, RevNumber(d_min(r, dr)));
    }
}

/// Pops a single leaf from the top of `candidates` and add those points to `hits`.
fn leaf_into_hits<I, U, D, C>(
    tree: &Tree<I, U, D, C>,
    query: &I,
    hits: &mut priority_queue::PriorityQueue<usize, OrdNumber<U>>,
    candidates: &mut priority_queue::PriorityQueue<&C, RevNumber<U>>,
) where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    let (leaf, RevNumber(d)) = candidates
        .pop()
        .unwrap_or_else(|| unreachable!("candidates is non-empty"));
    let distances = if leaf.is_singleton() {
        vec![d; leaf.indices().len()]
    } else {
        tree.data().query_to_many(query, &leaf.indices().collect::<Vec<_>>())
    };
    leaf.indices().zip(distances).for_each(|(i, d)| {
        hits.push(i, OrdNumber(d));
    });
}

/// Trims `hits` to contain only the k nearest neighbors.
fn trim_hits<U: Number>(k: usize, hits: &mut priority_queue::PriorityQueue<usize, OrdNumber<U>>) {
    while hits.len() > k {
        hits.pop()
            .unwrap_or_else(|| unreachable!("`hits` is non-empty and has at least k elements."));
    }
}
