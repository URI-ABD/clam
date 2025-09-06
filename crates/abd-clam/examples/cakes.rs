//! Example for running CAKES search.

use std::path::Path;

use abd_clam::{
    Tree,
    cakes::{KnnBfs, KnnDfs, KnnRrnn, ParSearch, RnnChess},
};
use ordered_float::OrderedFloat;
use rayon::prelude::*;

fn metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    distances::simd::euclidean_f32(a, b)
}

fn build_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    profi::prof!();
    metric(a, b)
}

fn knn_dfs_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    profi::prof!();
    metric(a, b)
}

fn knn_bfs_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    profi::prof!();
    metric(a, b)
}

fn knn_rrnn_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    profi::prof!();
    metric(a, b)
}

fn oracle_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    profi::prof!();
    metric(a, b)
}

fn read_data(name: &str) -> Result<[Vec<Vec<f32>>; 2], String> {
    let base = Path::new(".")
        .canonicalize()
        .map_err(|e| e.to_string())?
        .parent()
        .ok_or_else(|| "Failed to get parent directory.".to_string())?
        .join("data/ann_data");

    let data = read_array(base.join(format!("{}-train.npy", name)))?;
    let queries = read_array(base.join(format!("{}-test.npy", name)))?;

    Ok([data, queries])
}

fn read_array<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f32>>, String> {
    let array = ndarray_npy::read_npy::<_, ndarray::Array2<f32>>(path).map_err(|e| e.to_string())?;
    Ok(array.outer_iter().map(|row| row.to_vec()).collect())
}

fn search_tree<'a, M: Fn(&Vec<f32>, &Vec<f32>) -> f32 + Send + Sync>(tree: Tree<usize, Vec<f32>, f32, (), M>, queries: &[Vec<f32>], k: usize) {
    profi::prof!();

    // For each of the search algorithms, we change the metric to count distance computations separately.

    let tree = tree.with_metric(knn_dfs_metric);
    let dfs_results = KnnDfs(k).par_batch_search(&tree, queries);

    let tree = tree.with_metric(knn_bfs_metric);
    let bfs_results = KnnBfs(k).par_batch_search(&tree, queries);

    let tree = tree.with_metric(knn_rrnn_metric);
    let rrnn_results = KnnRrnn(k).par_batch_search(&tree, queries);

    let oracles = dfs_results
        .iter()
        .map(|res| res.iter().max_by_key(|&&(_, d)| OrderedFloat(d)).map(|&(_, radius)| RnnChess(radius)).unwrap())
        .collect::<Vec<_>>();

    let tree = tree.with_metric(oracle_metric);
    let oracle_results = oracles
        .into_par_iter()
        .zip(queries)
        .map(|(oracle, query)| oracle.par_search(&tree, query))
        .collect::<Vec<_>>();

    for (i, (res, oracle)) in dfs_results.into_iter().zip(oracle_results).enumerate() {
        assert_eq!(res.len(), k, "Query {i} expected {k} results, got {}", res.len());
        assert!(oracle.len() >= k, "Query {i} expected at least {k} oracle results, got {}", oracle.len());
    }

    core::mem::drop(bfs_results);
    core::mem::drop(rrnn_results);
}

fn main() -> Result<(), String> {
    let mut file = std::fs::File::create("./logs/profi-cakes.txt").map_err(|e| e.to_string())?;
    profi::print_on_exit!(to = &mut file);

    let k = 10;

    let [items, queries] = read_data("fashion-mnist")?;

    // Build tree with `build_metric` to count the number of distance computations in building the tree.
    let tree = Tree::par_new_minimal(items, build_metric)?;

    search_tree(tree, &queries[..1000], k);

    Ok(())
}
