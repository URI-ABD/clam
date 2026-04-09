//! Tests for the CAKES search algorithms.

use abd_clam::{
    DistanceValue, NamedAlgorithm, Tree,
    cakes::{self, Search},
    common_metrics,
};
use test_case::test_case;

mod common;

fn metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let d = common_metrics::euclidean(a, b);
    // Truncate to 3 decimal places for easier debugging
    (d * 1000.0).trunc() / 1000.0
}

#[test_case(10, 2; "10x2")]
#[test_case(100, 2; "100x2")]
#[test_case(1_000, 10; "1_000x10")]
fn vectors(car: usize, dim: usize) -> Result<(), String> {
    let data = common::data_gen::tabular(car, dim, -1.0, 1.0);
    let query = vec![0.0; dim];

    // Truncate all items to 3 decimal places for debugging
    let data = data
        .into_iter()
        .map(|v| v.into_iter().map(|x| (x * 1000.0).trunc() / 1000.0).collect::<Vec<f32>>())
        .collect::<Vec<_>>();

    let tree = Tree::new_minimal(data, metric)?;
    for radius in [0.5, 1.0, 1.5, 2.0] {
        let linear_alg = cakes::RnnLinear::new(radius);
        let linear_hits = linear_alg.search(&tree, &query);
        let linear_hits = sort_nondescending(linear_hits);

        let clustered_alg = cakes::RnnChess::new(radius);
        let clustered_hits = clustered_alg.search(&tree, &query);
        let clustered_hits = sort_nondescending(clustered_hits);
        check_hits(&linear_hits, &clustered_hits, &clustered_alg);

        let query_batch = vec![query.clone(); 10];
        let linear_hits = linear_alg.batch_search(&tree, &query_batch);
        let linear_hits = linear_hits.into_iter().map(sort_nondescending).collect::<Vec<_>>();

        let clustered_hits = clustered_alg.batch_search(&tree, &query_batch);
        let clustered_hits = clustered_hits.into_iter().map(sort_nondescending).collect::<Vec<_>>();
        for (linear_hits, clustered_hits) in linear_hits.iter().zip(clustered_hits.iter()) {
            check_hits(linear_hits, clustered_hits, &clustered_alg);
        }
    }

    for k in [1, 10, 100] {
        let linear_alg = cakes::KnnLinear::new(k);
        let linear_hits = linear_alg.search(&tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        assert_eq!(
            linear_hits.len(),
            k.min(car),
            "Not enough linear hits {} for k={}",
            linear_hits.len(),
            k.min(car)
        );

        let dfs_alg = cakes::KnnDfs::new(k);
        let dfs_hits = dfs_alg.search(&tree, &query);
        let dfs_hits = sort_nondescending(dfs_hits);
        check_hits(&linear_hits, &dfs_hits, &dfs_alg);

        let bfs_alg = cakes::KnnBfs::new(k);
        let bfs_hits = bfs_alg.search(&tree, &query);
        let bfs_hits = sort_nondescending(bfs_hits);
        check_hits(&linear_hits, &bfs_hits, &bfs_alg);

        let rrnn_alg = cakes::KnnRrnn::new(k);
        let rrnn_hits = rrnn_alg.search(&tree, &query);
        let rrnn_hits = sort_nondescending(rrnn_hits);
        check_hits(&linear_hits, &rrnn_hits, &rrnn_alg);

        let sieve_alg = cakes::KnnSieve::new(k);
        let sieve_hits = sieve_alg.search(&tree, &query);
        let sieve_hits = sort_nondescending(sieve_hits);
        check_hits(&linear_hits, &sieve_hits, &sieve_alg);
    }

    Ok(())
}

#[test_case(10_000, 10; "10_000x10")]
#[test_case(10_000, 100; "10_000x100")]
fn par_vectors(car: usize, dim: usize) -> Result<(), String> {
    let data = common::data_gen::tabular(car, dim, -1.0, 1.0);
    let query = vec![0.0; dim];

    let tree = Tree::par_new_minimal(data, metric)?;
    for radius in [1.0, 1.5, 2.0] {
        let linear_alg = cakes::RnnLinear::new(radius);
        let linear_hits = linear_alg.par_search(&tree, &query);
        let linear_hits = sort_nondescending(linear_hits);

        let clustered_alg = cakes::RnnChess::new(radius);
        let clustered_hits = clustered_alg.par_search(&tree, &query);
        let clustered_hits = sort_nondescending(clustered_hits);
        check_hits(&linear_hits, &clustered_hits, &clustered_alg);
    }

    for k in [1, 10, 100] {
        let linear_alg = cakes::KnnLinear::new(k);
        let linear_hits = linear_alg.par_search(&tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        assert_eq!(
            linear_hits.len(),
            k.min(car),
            "Not enough expected hits {} for k={}",
            linear_hits.len(),
            k.min(car)
        );

        let dfs_alg = cakes::KnnDfs::new(k);
        let dfs_hits = dfs_alg.par_search(&tree, &query);
        let dfs_hits = sort_nondescending(dfs_hits);
        check_hits(&linear_hits, &dfs_hits, &dfs_alg);

        let bfs_alg = cakes::KnnBfs::new(k);
        let bfs_hits = bfs_alg.par_search(&tree, &query);
        let bfs_hits = sort_nondescending(bfs_hits);
        check_hits(&linear_hits, &bfs_hits, &bfs_alg);

        let rrnn_alg = cakes::KnnRrnn::new(k);
        let rrnn_hits = rrnn_alg.par_search(&tree, &query);
        let rrnn_hits = sort_nondescending(rrnn_hits);
        check_hits(&linear_hits, &rrnn_hits, &rrnn_alg);

        let sieve_alg = cakes::KnnSieve::new(k);
        let sieve_hits = sieve_alg.par_search(&tree, &query);
        let sieve_hits = sort_nondescending(sieve_hits);
        check_hits(&linear_hits, &sieve_hits, &sieve_alg);
    }

    Ok(())
}

fn check_hits<T, Alg>(expected: &[(usize, T)], actual: &[(usize, T)], alg: &Alg)
where
    T: DistanceValue,
    Alg: NamedAlgorithm,
{
    assert_eq!(
        expected.len(),
        actual.len(),
        "{}: Hit count mismatch: \nexp {expected:?}, \ngot {actual:?}",
        &alg.name()
    );

    for (i, (&(_, e), &(_, a))) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(e, a, "{alg}: Distance mismatch at index {i}: \nexp {expected:?}, \ngot {actual:?}");
    }
}

fn sort_nondescending(mut items: Vec<(usize, f32)>) -> Vec<(usize, f32)> {
    items.sort_by(|&(_, l), (_, r)| l.total_cmp(r));
    items
}
