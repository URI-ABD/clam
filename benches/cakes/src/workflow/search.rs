//! Helpers for using the search modules from `abd_clam` in benchmarks.

use core::time::Duration;

use abd_clam::{
    cakes::{ParSearchAlgorithm, ParSearchable},
    cluster::ParCluster,
    metric::ParMetric,
};
use bench_utils::reports::CakesResults;
use distances::Number;

/// Run all the search algorithms on a cluster.
pub fn bench_all_algs<I, T, M, C, D>(
    report: &mut CakesResults<T>,
    metric: &M,
    queries: &[I],
    neighbors: Option<&[Vec<(usize, T)>]>,
    root: &C,
    data: &D,
    is_balanced: bool,
    is_permuted: bool,
    is_compressed: bool,
    max_time: Duration,
    ks: &[usize],
    radii: &[T],
    run_linear: bool,
    ranged_search: bool,
) where
    I: Send + Sync,
    T: Number,
    M: ParMetric<I, T>,
    C: ParCluster<T>,
    D: ParSearchable<I, T, C, M>,
{
    if ranged_search {
        for (i, &radius) in radii.iter().enumerate() {
            if i == 0 && run_linear {
                bench_algorithm(
                    report,
                    metric,
                    queries,
                    neighbors,
                    &abd_clam::cakes::RnnLinear(radius),
                    root,
                    data,
                    is_balanced,
                    is_permuted,
                    is_compressed,
                    max_time,
                );
            }

            bench_algorithm(
                report,
                metric,
                queries,
                neighbors,
                &abd_clam::cakes::RnnClustered(radius),
                root,
                data,
                is_balanced,
                is_permuted,
                is_compressed,
                max_time,
            );
        }
    }

    for (i, &k) in ks.iter().enumerate() {
        if i == 0 && run_linear {
            bench_algorithm(
                report,
                metric,
                queries,
                neighbors,
                &abd_clam::cakes::KnnLinear(k),
                root,
                data,
                is_balanced,
                is_permuted,
                is_compressed,
                max_time,
            );
        }

        bench_algorithm(
            report,
            metric,
            queries,
            neighbors,
            &abd_clam::cakes::KnnRepeatedRnn(k, T::ONE.double()),
            root,
            data,
            is_balanced,
            is_permuted,
            is_compressed,
            max_time,
        );

        bench_algorithm(
            report,
            metric,
            queries,
            neighbors,
            &abd_clam::cakes::KnnBreadthFirst(k),
            root,
            data,
            is_balanced,
            is_permuted,
            is_compressed,
            max_time,
        );

        bench_algorithm(
            report,
            metric,
            queries,
            neighbors,
            &abd_clam::cakes::KnnDepthFirst(k),
            root,
            data,
            is_balanced,
            is_permuted,
            is_compressed,
            max_time,
        );
    }
}

/// Run a single search algorithm on a cluster.
fn bench_algorithm<I, T, M, C, A, D>(
    report: &mut CakesResults<T>,
    metric: &M,
    queries: &[I],
    neighbors: Option<&[Vec<(usize, T)>]>,
    alg: &A,
    root: &C,
    data: &D,
    is_balanced: bool,
    is_permuted: bool,
    is_compressed: bool,
    max_time: Duration,
) where
    I: Send + Sync,
    T: Number,
    M: ParMetric<I, T>,
    C: ParCluster<T>,
    A: ParSearchAlgorithm<I, T, C, M, D>,
    D: ParSearchable<I, T, C, M>,
{
    let cluster_name = {
        let mut parts = Vec::with_capacity(3);
        if is_compressed {
            parts.push("Squishy");
        } else if is_permuted {
            parts.push("Permuted");
        }
        if is_balanced {
            parts.push("Balanced");
        }
        parts.push("Ball");
        parts.join("")
    };

    ftlog::info!("Running {} on {cluster_name} with {}...", alg.name(), data.name());
    let mut hits = Vec::with_capacity(100);
    let start = std::time::Instant::now();
    while start.elapsed() < max_time {
        hits.push(alg.par_batch_search(data, metric, root, queries));
    }
    let total_time = start.elapsed().as_secs_f32();

    let n_runs = queries.len() * hits.len();
    let time = total_time / n_runs.as_f32();
    let throughput = n_runs.as_f32() / total_time;
    ftlog::info!(
        "With {cluster_name}, Algorithm {} achieved Throughput {throughput} q/s",
        alg.name()
    );

    let last_hits = hits.last().unwrap_or_else(|| unreachable!("We ran it at least once"));
    let output_sizes = last_hits.iter().map(Vec::len).collect::<Vec<_>>();

    #[allow(clippy::option_if_let_else, clippy::branches_sharing_code)]
    let recalls = if let Some(_neighbors) = neighbors {
        // let mut recalls = Vec::with_capacity(neighbors.len());
        // for (i, neighbors) in neighbors.iter().enumerate() {
        //     let mut recall = Vec::with_capacity(neighbors.len());
        //     for (j, (idx, _)) in neighbors.iter().enumerate() {
        //         let mut count = 0;
        //         for hit in last_hits.iter() {
        //             if hit.iter().any(|(h, _)| *h == *idx) {
        //                 count += 1;
        //             }
        //         }
        //         recall.push(count.as_f32() / last_hits.len().as_f32());
        //     }
        //     recalls.push(recall);
        // }
        // recalls
        vec![1.0; queries.len()]
    } else {
        vec![1.0; queries.len()]
    };
    if let Some(radius) = alg.radius() {
        report.append_radial_result(
            &cluster_name,
            alg.name(),
            radius,
            time,
            throughput,
            &output_sizes,
            &recalls,
            0.0,
        );
    } else if let Some(k) = alg.k() {
        report.append_k_result(
            &cluster_name,
            alg.name(),
            k,
            time,
            throughput,
            &output_sizes,
            &recalls,
            0.0,
        );
    }
}
