//! Repeated RNN search, with increasing radii, for k-nearest neighbors.

use distances::Number;

use crate::{cakes::rnn::clustered, utils, Cluster, Tree};

use super::Hits;

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
    let mut radius = f64::EPSILON + tree.radius().as_f64() / tree.cardinality().as_f64();
    let [mut confirmed, mut straddlers] = clustered::tree_search(tree.data(), &tree.root, query, U::from(radius));

    let mut num_confirmed = count_hits(&confirmed);

    while num_confirmed == 0 {
        radius *= MULTIPLIER;
        [confirmed, straddlers] = clustered::tree_search(tree.data(), &tree.root, query, U::from(radius));
        num_confirmed = count_hits(&confirmed);
    }

    while num_confirmed < k {
        let lfd = utils::mean(
            &confirmed
                .iter()
                .chain(straddlers.iter())
                .map(|&(c, _)| c.lfd)
                .collect::<Vec<_>>(),
        );
        let factor = (k.as_f64() / num_confirmed.as_f64()).powf(1. / (lfd + f64::EPSILON));

        radius *= if factor < MULTIPLIER { factor } else { MULTIPLIER };
        [confirmed, straddlers] = clustered::tree_search(tree.data(), &tree.root, query, U::from(radius));
        num_confirmed = count_hits(&confirmed);
    }

    Hits::from_vec(
        k,
        clustered::leaf_search(&tree.data, confirmed, straddlers, query, U::from(radius)),
    )
    .extract()
}

/// Count the total cardinality of the clusters.
fn count_hits<T: Send + Sync + Copy, U: Number>(clusters: &[(&Cluster<T, U>, U)]) -> usize {
    clusters.iter().map(|(c, _)| c.cardinality).sum()
}

#[cfg(test)]
mod tests {

    use distances::vectors::euclidean;
    use symagen::random_data;

    use crate::{cakes::knn::linear, knn::tests::sort_hits, Cakes, PartitionCriteria, VecDataset};

    #[test]
    fn repeated_rnn() {
        let (cardinality, dimensionality) = (10_000, 100);
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

        for k in [1, 10, 100] {
            let repeated_nn = sort_hits(super::search(tree, query, k));
            assert_eq!(repeated_nn.len(), k);

            let linear_nn = sort_hits(linear::search(tree.data(), query, k, tree.indices()));
            assert_eq!(linear_nn, repeated_nn);
        }
    }
}
