//! Tests for Cakes.

use abd_clam::{knn, rnn, Cakes, Instance, PartitionCriteria, VecDataset};
use distances::Number;
use float_cmp::approx_eq;
use test_case::test_case;

mod utils;

#[test]
fn tiny() {
    let data = utils::gen_dataset_from(
        vec![vec![0., 0.], vec![1., 1.], vec![2., 2.], vec![3., 3.]],
        utils::euclidean,
        Some(vec![true, false, true, false]),
    );
    let criteria = PartitionCriteria::default();
    let cakes = Cakes::new(data, None, &criteria);

    let query = vec![0., 1.];
    let (results, _): (Vec<_>, Vec<_>) = cakes
        .rnn_search(&query, 1.5, rnn::Algorithm::Clustered)
        .into_iter()
        .unzip();
    assert_eq!(results.len(), 2);

    let result_points = results.iter().map(|&i| &cakes[i]).collect::<Vec<_>>();
    assert!(result_points.contains(&&vec![0., 0.]));
    assert!(result_points.contains(&&vec![1., 1.]));

    let query = vec![1., 1.];
    let (results, _): (Vec<_>, Vec<_>) = cakes
        .rnn_search(&query, 0., rnn::Algorithm::Clustered)
        .into_iter()
        .unzip();
    assert_eq!(results.len(), 1);

    assert!(results.iter().map(|&i| &cakes[i]).any(|x| x == [1., 1.].as_slice()));
}

#[test]
fn line() {
    let data = (-100..=100).map(|x| vec![x.as_f32()]).collect::<Vec<_>>();
    let metadata = Some(data.iter().map(|x| x[0] > 0.0).collect());
    let data = utils::gen_dataset_from(data, utils::euclidean, metadata);
    let criteria = PartitionCriteria::default();
    let cakes = Cakes::new(data, Some(42), &criteria);

    let queries = (-10..=10).step_by(2).map(|x| vec![x.as_f32()]).collect::<Vec<_>>();
    for v in [2, 10, 50] {
        let radius = v.as_f32();
        let n_hits = 1 + 2 * v;

        for (i, query) in queries.iter().enumerate() {
            let linear_hits = cakes.rnn_search(query, radius, rnn::Algorithm::Linear);
            assert_eq!(
                n_hits,
                linear_hits.len(),
                "Failed linear search: query: {i}, radius: {radius}, linear: {linear_hits:?}",
            );

            let ranged_hits = cakes.rnn_search(query, radius, rnn::Algorithm::Clustered);
            assert_eq!(
                n_hits,
                ranged_hits.len(),
                "Failed Clustered search: query: {i}, radius: {radius}, ranged: {ranged_hits:?}",
            );

            let recall = utils::compute_recall(ranged_hits, linear_hits);
            assert!(approx_eq!(f32, recall, 1.0), "Clustered Recall: {}", recall);
        }
    }
}

#[ignore = "Fails with Sieve and SieveSepCenter."]
#[test_case(1000, 10; "1k_10")]
#[test_case(1000, 100; "1k_100")]
#[test_case(10_000, 10; "10k_10")]
#[test_case(10_000, 100; "10k_100")]
fn vectors(cardinality: usize, dimensionality: usize) {
    let seed = 42;

    let data = utils::gen_dataset(cardinality, dimensionality, seed, utils::euclidean);
    let cakes = Cakes::new(data, Some(seed), &PartitionCriteria::default());

    let num_queries = 100;
    let queries = utils::gen_dataset(num_queries, dimensionality, seed, utils::euclidean);
    let queries = (0..num_queries).map(|i| &queries[i]).collect::<Vec<_>>();

    let radii = (1..3).rev().map(|i| 10_f32.powi(-i)).collect::<Vec<_>>();
    let ks = (1..3).map(|i| 10usize.pow(i)).collect::<Vec<_>>();
    check_search_quality(&queries, &cakes, &radii, &ks);
}

#[ignore = "Fails with RepeatedRnn."]
#[test_case(1000, "ACTG", utils::hamming; "100_ACTG_Ham")]
#[test_case(1000, "ACTG", utils::levenshtein; "100_ACTG_Lev")]
#[test_case(1000, "ACTG", utils::needleman_wunsch; "100_ACTG_NW")]
fn strings(cardinality: usize, alphabet: &str, metric: fn(&String, &String) -> u16) {
    let seed = 42;
    let seq_len = 100;

    let data = symagen::random_data::random_string(cardinality, seq_len, seq_len, alphabet, seed);

    let data = VecDataset::<_, _, bool>::new("test".to_string(), data.clone(), metric, false, None);
    let cakes = Cakes::new(data, Some(42), &PartitionCriteria::default());

    let num_queries = 10;
    let queries = symagen::random_data::random_string(num_queries, seq_len, seq_len, alphabet, seed + 1);
    let queries = (0..num_queries).map(|i| &queries[i]).collect::<Vec<_>>();

    check_search_quality(&queries, &cakes, &[1, 5, 10], &[1, 5, 10]);
}

fn check_search_quality<I: Instance, U: Number, M: Instance>(
    queries: &[&I],
    cakes: &Cakes<I, U, VecDataset<I, U, M>>,
    radii: &[U],
    ks: &[usize],
) {
    for &radius in radii {
        for (i, &query) in queries.iter().enumerate() {
            let linear_hits = cakes.rnn_search(query, radius, rnn::Algorithm::Linear);
            let ranged_hits = cakes.rnn_search(query, radius, rnn::Algorithm::Clustered);

            if linear_hits.is_empty() {
                assert!(
                    ranged_hits.is_empty(),
                    "Linear search returned no hits, but Clustered search returned some."
                );
            } else {
                assert_eq!(
                    linear_hits.len(),
                    ranged_hits.len(),
                    "Incorrect number of hits: query: {i}, radius: {radius}, linear: {}, ranged: {}",
                    linear_hits.len(),
                    ranged_hits.len(),
                );

                let recall = utils::compute_recall(ranged_hits, linear_hits);
                assert!(approx_eq!(f32, recall, 1.0), "Clustered Recall: {}", recall);
            }
        }
    }

    for &k in ks {
        for (i, query) in queries.iter().enumerate() {
            let linear_hits = cakes.knn_search(query, k, knn::Algorithm::Linear);

            for &variant in knn::Algorithm::variants() {
                let ranged_hits = cakes.knn_search(query, k, variant);

                if linear_hits.is_empty() {
                    assert!(
                        ranged_hits.is_empty(),
                        "Linear search returned no hits, but {} search returned some.",
                        variant.name()
                    );
                } else {
                    assert_eq!(
                        linear_hits.len(),
                        ranged_hits.len(),
                        "Incorrect number of hits: query: {i}, k: {k}, linear: {}, {}: {}",
                        linear_hits.len(),
                        variant.name(),
                        ranged_hits.len(),
                    );

                    let recall = utils::compute_recall(ranged_hits, linear_hits.clone());
                    assert!(
                        approx_eq!(f32, recall, 1.0),
                        "query: {i}, k: {k}, {}: {}",
                        variant.name(),
                        recall
                    );
                }
            }
        }
    }
}

#[test_case(10)]
#[test_case(100)]
fn get_trees(num_shards: u64) {
    let data = utils::gen_dataset(1000, 10, 42, utils::euclidean);

    let criteria = PartitionCriteria::default();
    let cakes = Cakes::new(data, None, &criteria);

    let shards = cakes.shards();
    assert_eq!(shards.len(), 1);

    let trees = cakes.trees();
    assert_eq!(trees.len(), 1);

    let shards = (0..num_shards)
        .map(|i| utils::gen_dataset(100, 10, i, utils::euclidean))
        .collect();

    let cakes = Cakes::new_randomly_sharded(shards, None, &criteria);

    let shards = cakes.shards();
    assert_eq!(shards.len(), num_shards as usize);

    let trees = cakes.trees();
    assert_eq!(trees.len(), num_shards as usize);
}

#[test]
fn save_load_single() {
    let data = utils::gen_dataset(1000, 10, 42, utils::euclidean);

    let criteria = PartitionCriteria::default();
    let cakes = Cakes::new(data, None, &criteria);

    let tmp_dir = tempdir::TempDir::new("cakes-test").unwrap();
    cakes.save(tmp_dir.path()).unwrap();

    let cakes = Cakes::<Vec<f32>, f32, VecDataset<_, _, bool>>::load(tmp_dir.path(), utils::euclidean, false).unwrap();

    let shards = cakes.shards();
    assert_eq!(shards.len(), 1);

    let trees = cakes.trees();
    assert_eq!(trees.len(), 1);
}

#[test_case(10)]
#[test_case(100)]
fn save_load_sharded(num_shards: u64) {
    let shards = (0..num_shards)
        .map(|i| utils::gen_dataset(100, 10, i, utils::euclidean))
        .collect();

    let criteria = PartitionCriteria::default();
    let cakes = Cakes::new_randomly_sharded(shards, None, &criteria);

    let tmp_dir = tempdir::TempDir::new("sharded-cakes-test").unwrap();
    cakes.save(tmp_dir.path()).unwrap();

    let cakes = Cakes::<Vec<f32>, f32, VecDataset<_, _, bool>>::load(tmp_dir.path(), utils::euclidean, false).unwrap();

    let shards = cakes.shards();
    assert_eq!(shards.len(), num_shards as usize);

    let trees = cakes.trees();
    assert_eq!(trees.len(), num_shards as usize);
}
