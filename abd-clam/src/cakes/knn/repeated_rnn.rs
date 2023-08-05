//! Repeated RNN search, with increasing radii, for k-nearest neighbors.

use core::{cmp::Ordering, f64::EPSILON};

use distances::Number;

use crate::{cakes::rnn::clustered::tree_search, utils, Tree};

/// The multiplier to use for increasing the radius in the repeated RNN algorithm.
const MULTIPLIER: f64 = 2.0;

/// K-Nearest Neighbor search using a repeated RNN search.
///
/// # Arguments
///
/// * `tree` - The tree to search.
/// * `query` - The query to search around.
/// * `k` - The number of neighbors to search for.
///
/// # Returns
///
/// A vector of 2-tuples, where the first element is the index of the instance
/// and the second element is the distance from the query to the instance.
pub fn search<T, U, D>(tree: &Tree<T, U, D>, query: T, k: usize) -> Vec<(usize, U)>
where
    T: Send + Sync + Copy,
    U: Number,
    D: crate::Dataset<T, U>,
{
    let mut radius = EPSILON + tree.radius().as_f64() / tree.cardinality().as_f64();
    let [mut confirmed, mut straddlers] = tree_search(tree.data(), tree.root(), query, U::from(radius));

    let mut num_hits = confirmed
        .iter()
        .chain(straddlers.iter())
        .map(|&(c, _)| c.cardinality)
        .sum::<usize>();

    while num_hits == 0 {
        radius *= MULTIPLIER;
        [confirmed, straddlers] = tree_search(tree.data(), tree.root(), query, U::from(radius));
        num_hits = confirmed
            .iter()
            .chain(straddlers.iter())
            .map(|&(c, _)| c.cardinality)
            .sum::<usize>();
    }

    while num_hits < k {
        let lfd = utils::mean(
            &confirmed
                .iter()
                .chain(straddlers.iter())
                .map(|&(c, _)| c.lfd)
                .collect::<Vec<_>>(),
        );
        let factor = (k.as_f64() / num_hits.as_f64()).powf(1. / (lfd + EPSILON));

        radius *= if factor < MULTIPLIER { factor } else { MULTIPLIER };
        [confirmed, straddlers] = tree_search(tree.data(), tree.root(), query, U::from(radius));
        num_hits = confirmed
            .iter()
            .chain(straddlers.iter())
            .map(|&(c, _)| c.cardinality)
            .sum::<usize>();
    }

    let mut hits = confirmed
        .into_iter()
        .chain(straddlers.into_iter())
        .flat_map(|(c, d)| {
            let indices = c.indices(tree.data());
            let distances = if c.is_singleton() {
                vec![d; c.cardinality]
            } else {
                tree.data().query_to_many(query, indices)
            };
            indices.iter().copied().zip(distances.into_iter())
        })
        .collect::<Vec<_>>();

    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    hits[..k].to_vec()
}

#[cfg(test)]
mod tests {

    use distances::vectors::euclidean;
    use symagen::random_data;

    use crate::{cakes::knn::linear, Cakes, PartitionCriteria, VecDataset};

    #[test]
    fn repeated_rnn() {
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
            let linear_nn = linear::search(tree.data(), query, k, tree.indices());
            let repeated_nn = super::search(tree, query, k);

            assert_eq!(linear_nn, repeated_nn);
        }
    }
}
