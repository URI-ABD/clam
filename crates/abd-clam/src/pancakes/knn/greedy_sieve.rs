//! K-NN search with greedy sieve.

use crate::{pancakes::CodecData, pancakes::SquishyBall, Cluster, Instance};
use distances::number::UInt;

use crate::cakes::knn::greedy_sieve::d_min;
use crate::cakes::knn::OrdNumber;
use crate::cakes::knn::RevNumber;

/// Searches for the k nearest neighbors of `query` in `data` using the greedy sieve algorithm.
///
/// # Arguments
///
/// * `query` - The query point.
/// * `k` - The number of nearest neighbors to find.
/// * `data` - The data structure to search.
///
/// # Returns
///
/// A vector of pairs `(i, u)` where `i` is the index of a point in the data structure and `u` is the distance
/// from `query` to that point.
pub fn search<I, U, M>(query: &I, k: usize, data: &CodecData<I, U, M>) -> Vec<(usize, U)>
where
    I: Instance,
    U: UInt,
    M: Instance,
{
    let mut candidates = priority_queue::PriorityQueue::<&SquishyBall<U>, RevNumber<U>>::new();
    let mut hits = priority_queue::PriorityQueue::<usize, OrdNumber<U>>::new();

    let root = data.root();
    let centers = &data.centers();
    let root_center = &centers[&root.arg_center()];
    let d = data.metric()(root_center, query);

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
        pop_till_leaf(data, query, &mut candidates);
        leaf_into_hits(data, query, &mut hits, &mut candidates);
        trim_hits(k, &mut hits);
    }

    todo!();
}

/// Pops from the top of `candidates` until the top candidate is a leaf cluster.
fn pop_till_leaf<I, U, M>(
    data: &CodecData<I, U, M>,
    query: &I,
    candidates: &mut priority_queue::PriorityQueue<&SquishyBall<U>, RevNumber<U>>,
) where
    I: Instance,
    U: UInt,
    M: Instance,
{
    while !candidates
        .peek()
        .map_or_else(|| unreachable!("`candidates` is non-empty"), |(c, _)| c.squish())
    {
        let [l, r] = candidates.pop().map_or_else(
            || unreachable!("`candidates` is non-empty"),
            |(c, _)| {
                (*c).children()
                    .unwrap_or_else(|| unreachable!("elements are non-leaves"))
            },
        );
        let l_center = &data.centers()[&l.arg_center()];
        let r_center = &data.centers()[&r.arg_center()];

        let [dl, dr] = [data.metric()(l_center, query), data.metric()(r_center, query)];
        candidates.push(l, RevNumber(d_min(l, dl)));
        candidates.push(r, RevNumber(d_min(r, dr)));
    }
}

/// Pops a single leaf from the top of `candidates` and pushes it into `hits`.
fn leaf_into_hits<I, U, M>(
    data: &CodecData<I, U, M>,
    query: &I,
    hits: &mut priority_queue::PriorityQueue<usize, OrdNumber<U>>,
    candidates: &mut priority_queue::PriorityQueue<&SquishyBall<U>, RevNumber<U>>,
) where
    I: Instance,
    U: UInt,
    M: Instance,
{
    let (leaf, RevNumber(d)) = candidates
        .pop()
        .unwrap_or_else(|| unreachable!("candidates is non-empty"));
    let distances = if leaf.is_singleton() {
        vec![
            d;
            data.load_leaf_data(leaf)
                .unwrap_or_else(|e| unreachable!("Impossible by construction: {e}"))
                .len()
        ]
    } else {
        let points = data
            .load_leaf_data(leaf)
            .unwrap_or_else(|e| unreachable!("Impossible by construction.: {e}"));

        points.into_iter().map(|point| data.metric()(query, &point)).collect()
    };
    leaf.indices().zip(distances).for_each(|(i, d)| {
        hits.push(i, OrdNumber(d));
    });
}

/// Trims `hits` to contain only the k nearest neighbors.
fn trim_hits<U: UInt>(k: usize, hits: &mut priority_queue::PriorityQueue<usize, OrdNumber<U>>) {
    while hits.len() > k {
        hits.pop()
            .unwrap_or_else(|| unreachable!("`hits` is non-empty and has at least k elements."));
    }
}
