//! Tests for the RNN-search algorithms.

use abd_clam::{knn, rnn, PartitionCriteria, Tree};
use distances::Number;
use float_cmp::assert_approx_eq;
use test_case::test_case;

mod utils;

#[test]
fn linear() {
    let data = (-10..=10).map(|i| vec![i.as_f32()]).collect::<Vec<_>>();
    let metadata = data.iter().map(|i| i[0] > 0.0).collect::<Vec<_>>();
    let data = utils::gen_dataset_from(data, utils::euclidean, Some(metadata));

    let query = &vec![0.0];

    let criteria = PartitionCriteria::default();
    let tree = Tree::new(data, None).partition(&criteria);

    let linear_knn = knn::Algorithm::Linear.search(&tree, query, 3);
    let linear_rnn = rnn::Algorithm::Linear.search(query, 1.5, &tree);

    for hits in [linear_knn, linear_rnn] {
        assert_eq!(hits.len(), 3);

        let distances = {
            let mut distances = hits.iter().map(|(_, d)| *d).collect::<Vec<_>>();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            distances
        };
        let true_distances = vec![0.0, 1.0, 1.0];

        assert_eq!(distances, true_distances);
    }
}

#[test_case(1000, 10; "1k_10")]
#[test_case(1000, 100; "1k_100")]
#[test_case(10_000, 10; "10k_10")]
#[test_case(10_000, 100; "10k_100")]
#[test_case(100_000, 10; "100k_10")]
#[test_case(100_000, 100; "100k_100")]
fn variants(cardinality: usize, dimensionality: usize) {
    let seed = 42;

    let data = utils::gen_dataset(cardinality, dimensionality, seed, utils::euclidean);
    let query = &vec![0.; dimensionality];

    let criteria = PartitionCriteria::default();
    let tree = Tree::new(data, None).partition(&criteria);

    for k in (0..3).map(|i| 10_usize.pow(i)) {
        let linear_nn = knn::Algorithm::Linear.search(&tree, query, k);
        assert_eq!(linear_nn.len(), k);

        for variant in knn::Algorithm::variants() {
            let variant_nn = variant.search(&tree, query, k);

            assert_eq!(linear_nn.len(), variant_nn.len());

            let recall = utils::compute_recall(linear_nn.clone(), variant_nn);
            assert_approx_eq!(f32, recall, 1.0);
        }
    }

    for radius in (1..=10).rev().map(|i| 10_f32.powi(-i)) {
        let linear_nn = rnn::Algorithm::Linear.search(query, radius, &tree);

        for variant in rnn::Algorithm::variants() {
            let variant_nn = variant.search(query, radius, &tree);

            if linear_nn.is_empty() {
                assert!(
                    variant_nn.is_empty(),
                    "Linear search returned no neighbors, but {} search returned some.",
                    variant.name()
                );
                continue;
            }

            assert_eq!(
                variant_nn.len(),
                linear_nn.len(),
                "Linear search returned {} neighbors, but {} search returned {}.",
                linear_nn.len(),
                variant.name(),
                variant_nn.len()
            );

            let recall = utils::compute_recall(linear_nn.clone(), variant_nn);
            assert_approx_eq!(f32, recall, 1.0);
        }

        if linear_nn.len() > 100 {
            break;
        }
    }
}
