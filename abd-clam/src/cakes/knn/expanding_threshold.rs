//! Search function and helper functions for knn with expanding threshold.

use distances::Number;

use crate::{Cluster, Dataset, Tree};

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
pub fn search<T, U, D>(tree: &Tree<T, U, D>, query: T, k: usize) -> Vec<(usize, U)>
where
    T: Send + Sync + Copy,
    U: Number,
    D: Dataset<T, U>,
{
    let mut candidates = priority_queue::PriorityQueue::<&Cluster<T, U>, RevNumber<U>>::new();
    let mut hits = priority_queue::PriorityQueue::<usize, OrdNumber<U>>::new();
    let d = tree.root().distance_to_instance(tree.data(), query);
    candidates.push(tree.root(), RevNumber(d_min(tree.root(), d)));

    // stop if we have enough hits and the farthest hit is closer than the closest cluster by delta_min.
    while hits.len() < k
        || (!candidates.is_empty()
            && (hits.peek().unwrap_or_else(|| unreachable!("`hits` is non-empty")).1 .0
                >= candidates
                    .peek()
                    .unwrap_or_else(|| unreachable!("`candidates` is non-empty"))
                    .1
                     .0))
    {
        pop_till_leaf(tree, query, &mut candidates);
        leaf_into_hits(tree, query, &mut hits, &mut candidates);
        trim_hits(k, &mut hits);
    }
    assert!(hits.len() >= k);

    hits.into_iter().map(|(i, OrdNumber(d))| (i, d)).collect()
}

/// Calculates the theoretical best case distance for a point in a cluster, i.e.,
/// the closest a point in a given cluster could possibly be to the query.
fn d_min<T: Send + Sync + Copy, U: Number>(c: &Cluster<T, U>, d: U) -> U {
    if d < c.radius {
        U::zero()
    } else {
        d - c.radius
    }
}

/// Pops from the top of `candidates` until the top candidate is a leaf cluster.
fn pop_till_leaf<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>>(
    tree: &Tree<T, U, D>,
    query: T,
    candidates: &mut priority_queue::PriorityQueue<&Cluster<T, U>, RevNumber<U>>,
) {
    while !candidates
        .peek()
        .unwrap_or_else(|| unreachable!("`candidates` is non-empty"))
        .0
        .is_leaf()
    {
        let [l, r] = candidates
            .pop()
            .unwrap_or_else(|| unreachable!("`candidates` is non-empty"))
            .0
            .children()
            .unwrap_or_else(|| unreachable!("elements are non-leaves"));
        let [dl, dr] = [
            l.distance_to_instance(tree.data(), query),
            r.distance_to_instance(tree.data(), query),
        ];
        candidates.push(l, RevNumber(d_min(l, dl)));
        candidates.push(r, RevNumber(d_min(r, dr)));
    }
}

/// Pops a single leaf from the top of candidates and add those points to hits.
fn leaf_into_hits<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>>(
    tree: &Tree<T, U, D>,
    query: T,
    hits: &mut priority_queue::PriorityQueue<usize, OrdNumber<U>>,
    candidates: &mut priority_queue::PriorityQueue<&Cluster<T, U>, RevNumber<U>>,
) {
    let (leaf, RevNumber(d)) = candidates
        .pop()
        .unwrap_or_else(|| unreachable!("candidates is non-empty"));
    let is = leaf.indices(tree.data());
    let ds = if leaf.is_singleton() {
        vec![d; is.len()]
    } else {
        tree.data().query_to_many(query, is)
    };
    is.iter().zip(ds.into_iter()).for_each(|(&i, d)| {
        hits.push(i, OrdNumber(d));
    });
}

/// Trims hits to contain only the k-nearest neighbors.
fn trim_hits<U: Number>(k: usize, hits: &mut priority_queue::PriorityQueue<usize, OrdNumber<U>>) {
    while hits.len() > k {
        hits.pop()
            .unwrap_or_else(|| unreachable!("`hits` is non-empty and has at least k elements."));
    }
}

#[cfg(test)]
mod tests {

    use distances::vectors::euclidean;
    use symagen::random_data;

    use crate::{cakes::knn::linear, knn::tests::sort_hits, Cakes, PartitionCriteria, VecDataset};

    #[test]
    fn expanding_thresholds() {
        let (cardinality, dimensionality) = (1_000, 10);
        let (min_val, max_val) = (-1.0, 1.0);
        let seed = 42;

        let data = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);
        let data = data.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let data = VecDataset::new("knn-test".to_string(), data, euclidean::<_, f32>, false);

        let query = random_data::random_f32(1, dimensionality, min_val, max_val, seed * 2);
        let query = query[0].as_slice();

        let criteria = PartitionCriteria::default();
        let model = Cakes::new(data, Some(seed), criteria);
        let tree = model.tree();

        for k in [100, 10, 1] {
            let linear_nn = sort_hits(linear::search(tree.data(), query, k, tree.indices()));
            let thresholds_nn = sort_hits(super::search(tree, query, k));
            assert_eq!(linear_nn, thresholds_nn);
        }
    }
}
