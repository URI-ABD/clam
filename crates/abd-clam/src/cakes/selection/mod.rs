//! Utilities for selecting the fastest CAKES KNN search algorithm for a given dataset and metric.

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    cakes::{Cakes, ParSearch, Search},
};

/// Measures the throughput (Queries per Second) of a CAKES algorithm on the given root cluster with the given metric.
///
/// This function runs the algorithm on the provided queries and measures the time taken to complete them. It uses only
/// a single thread for the measurement.
#[allow(clippy::cast_precision_loss, clippy::while_float)]
pub fn measure_throughput<Id, I, T, A, M, Alg>(tree: &Tree<Id, I, T, A, M>, n_queries: usize, alg: &Alg, min_time_secs: f64) -> f64
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Alg: Search<Id, I, T, A, M> + ?Sized,
{
    let n_queries = n_queries.min(tree.cardinality());
    let queries = tree.items[..n_queries].iter().map(|(_, item)| item).collect::<Vec<_>>();

    let mut total_queries = 0;
    let min_time = std::time::Duration::from_secs_f64(min_time_secs);

    let start = std::time::Instant::now();
    while start.elapsed() < min_time {
        let _results = queries.iter().map(|query| alg.search(tree, query)).collect::<Vec<_>>();
        total_queries += queries.len();
    }
    total_queries as f64 / start.elapsed().as_secs_f64()
}

/// Parallel version of [`measure_throughput`].
#[allow(clippy::cast_precision_loss, clippy::while_float)]
pub fn par_measure_throughput<Id, I, T, A, M, Alg>(tree: &Tree<Id, I, T, A, M>, n_queries: usize, alg: &Alg, min_time_secs: f64) -> f64
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    Alg: ParSearch<Id, I, T, A, M> + ?Sized,
{
    let n_queries = n_queries.min(tree.cardinality());
    let queries = tree.items[..n_queries].iter().map(|(_, item)| item).collect::<Vec<_>>();

    let mut total_queries = 0;
    let min_time = std::time::Duration::from_secs_f64(min_time_secs);

    let start = std::time::Instant::now();
    while start.elapsed() < min_time {
        let _results = queries.par_iter().map(|query| alg.par_search(tree, query)).collect::<Vec<_>>();
        total_queries += queries.len();
    }
    total_queries as f64 / start.elapsed().as_secs_f64()
}

/// Selects the fastest CAKES algorithm for the given dataset and metric.
#[allow(clippy::type_complexity)]
pub fn select_fastest_algorithm<'a, Id, I, T, A, M>(
    tree: &Tree<Id, I, T, A, M>,
    n_queries: usize,
    min_time_secs: f64,
    algorithms: &'a [Cakes<T>],
) -> (&'a Cakes<T>, f64)
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    algorithms
        .iter()
        .map(|alg| {
            let throughput = measure_throughput(tree, n_queries, alg, min_time_secs);
            (alg, throughput)
        })
        .max_by_key(|(_, throughput)| crate::utils::MaxItem((), *throughput))
        .unwrap_or_else(|| unreachable!("We created more than zero algorithms"))
}

/// Parallel version of [`select_fastest_algorithm`](.
#[allow(clippy::type_complexity)]
pub fn par_select_fastest_algorithm<'a, Id, I, T, A, M>(
    tree: &Tree<Id, I, T, A, M>,
    n_queries: usize,
    min_time_secs: f64,
    algorithms: &[&'a dyn ParSearch<Id, I, T, A, M>],
) -> (&'a dyn ParSearch<Id, I, T, A, M>, f64)
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    algorithms
        .iter()
        .map(|&alg| {
            let throughput = par_measure_throughput(tree, n_queries, alg, min_time_secs);
            (alg, throughput)
        })
        .max_by_key(|(_, throughput)| crate::utils::MaxItem((), *throughput))
        .unwrap_or_else(|| unreachable!("We created more than zero algorithms"))
}
