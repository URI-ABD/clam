//! Tests for the dataset module.

use abd_clam::{Dataset, VecDataset};
use rand::prelude::*;
use tempdir::TempDir;
use test_case::test_case;

mod utils;

#[test]
fn reordering() {
    let cardinality = 10_000;
    let dimensionality = 10;

    for i in 0..10 {
        let reference_data =
            symagen::random_data::random_tabular_seedable::<u32>(cardinality, dimensionality, 0, 100_000, i);
        let metadata = reference_data.iter().map(|x| x[0] > 50_000).collect::<Vec<_>>();
        for _ in 0..10 {
            let mut dataset = VecDataset::new(
                format!("test-{i}"),
                reference_data.clone(),
                utils::euclidean_sq,
                false,
                Some(metadata.clone()),
            );
            let mut new_indices = (0..cardinality).collect::<Vec<_>>();
            new_indices.shuffle(&mut rand::thread_rng());

            dataset.permute_instances(&new_indices).unwrap();
            for i in 0..cardinality {
                assert_eq!(dataset[i], reference_data[new_indices[i]]);
            }
        }
    }
}

#[test]
fn original_indices() {
    let data = (1_u32..7).map(|x| vec![x * 2]).collect::<Vec<_>>();
    let permutation = vec![1, 3, 4, 0, 5, 2];
    let permuted_data = permutation.iter().map(|&i| data[i].clone()).collect::<Vec<_>>();
    // let permuted_data = vec![vec![4], vec![8], vec![10], vec![2], vec![12], vec![6]];

    let mut dataset = VecDataset::<_, _, bool>::new("test".to_string(), data, utils::euclidean_sq, false, None);
    dataset.permute_instances(&permutation).unwrap();

    assert_eq!(dataset.data(), permuted_data);

    for (i, (p, v)) in permutation.into_iter().zip(permuted_data).enumerate() {
        assert_eq!(dataset.original_index(i), p);
        assert_eq!(dataset[i], v);
    }
}

#[test]
fn save_load_tiny() {
    let metric = utils::euclidean_sq::<u32>;
    let mut data = utils::gen_dataset_from(vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10]], metric, None);

    let indices = (0..data.cardinality()).rev().collect::<Vec<_>>();
    data.permute_instances(&indices).unwrap();

    let tmp_dir = TempDir::new("save_load_deterministic").unwrap();
    let tmp_file = tmp_dir.path().join("dataset.save");
    data.save(&tmp_file).unwrap();

    let other = VecDataset::<_, _, bool>::load(&tmp_file, metric, false).unwrap();

    assert_eq!(other.data(), data.data());
    assert_eq!(other.permuted_indices(), data.permuted_indices());
    assert_eq!(data.cardinality(), other.cardinality());
}

#[test_case(1000, 10; "1k_10")]
#[test_case(1000, 100; "1k_100")]
#[test_case(10_000, 10; "10k_10")]
#[test_case(10_000, 100; "10k_100")]
fn save_load(cardinality: usize, dimensionality: usize) {
    let metric = utils::euclidean_sq::<u32>;
    let tmp_dir = TempDir::new("save_load_deterministic").unwrap();

    for i in 0..5 {
        let reference_data =
            symagen::random_data::random_tabular_seedable::<u32>(cardinality, dimensionality, 0, 100_000, i);
        let tmp_file = tmp_dir.path().join(format!("dataset_{}.save", i));

        let mut dataset = VecDataset::<_, _, bool>::new("test".to_string(), reference_data, metric, false, None);
        if i % 2 == 0 {
            let indices = (0..dataset.cardinality()).rev().collect::<Vec<_>>();
            dataset.permute_instances(&indices).unwrap();
        }
        dataset.save(&tmp_file).unwrap();

        let other = VecDataset::<Vec<u32>, u32, bool>::load(&tmp_file, metric, false).unwrap();

        assert_eq!(other.data(), dataset.data());
        assert_eq!(other.name(), dataset.name());
        assert_eq!(other.permuted_indices(), dataset.permuted_indices());
        assert_eq!(dataset.cardinality(), other.cardinality());
    }
}

#[test]
fn load_errors() {
    // TODO: Expand this test to check other error conditions.

    let data = vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10]];
    let tmp_dir = TempDir::new("save_load_deterministic").unwrap();
    let tmp_file = tmp_dir.path().join("dataset.save");

    // Construct it with u32
    let mut dataset = VecDataset::<_, _, bool>::new("test".to_string(), data, utils::euclidean_sq, false, None);
    let indices = (0..dataset.cardinality()).rev().collect::<Vec<_>>();
    dataset.permute_instances(&indices).unwrap();
    dataset.save(&tmp_file).unwrap();

    // Try to load it back in as f32
    let other = VecDataset::<Vec<f32>, f32, bool>::load(&tmp_file, utils::euclidean, false);
    assert!(other.is_err());
}
