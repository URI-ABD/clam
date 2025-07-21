//! Steps in the workflow of running CAKES benchmarks.
#![allow(
    clippy::cognitive_complexity,
    clippy::fn_params_excessive_bools,
    clippy::too_many_arguments,
    clippy::missing_errors_doc
)]

use core::time::Duration;

use abd_clam::{cakes::PermutedBall, Ball, Cluster, Dataset, FlatVec, ParDiskIO};
use bench_utils::{reports::CakesResults, Complex};
use distances::Number;
use rand::prelude::*;

use crate::metric::ParCountingMetric;

pub mod search;
pub mod trees;

/// Run the workflow of the CAKES benchmarks on a fasta dataset.
pub fn run_radio_ml<P: AsRef<std::path::Path>, M: ParCountingMetric<Vec<Complex<f64>>, f64>>(
    out_dir: &P,
    data: &FlatVec<Vec<Complex<f64>>, usize>,
    metric: &M,
    queries: &[Vec<Complex<f64>>],
    radial_fractions: &[f32],
    ks: &[usize],
    seed: Option<u64>,
    max_time: Duration,
    run_linear: bool,
    balanced: bool,
    permuted: bool,
    ranged_search: bool,
    rebuild_trees: bool,
) -> Result<(), String> {
    let all_paths = trees::AllPaths::new(out_dir, data.name());
    if rebuild_trees || !all_paths.all_exist(balanced, permuted, false) {
        trees::build_all(out_dir, data, metric, seed, permuted, balanced, None)?;
    }
    run::<_, _, _, usize>(
        &all_paths,
        metric,
        queries,
        None,
        radial_fractions,
        ks,
        max_time,
        run_linear,
        balanced,
        permuted,
        ranged_search,
    )
}

/// Run the workflow of the CAKES benchmarks on a fasta dataset.
pub fn run_fasta<P: AsRef<std::path::Path>, M: ParCountingMetric<String, u32>>(
    out_dir: &P,
    data: &FlatVec<String, String>,
    metric: &M,
    queries: &[String],
    radial_fractions: &[f32],
    ks: &[usize],
    seed: Option<u64>,
    max_time: Duration,
    run_linear: bool,
    balanced: bool,
    permuted: bool,
    ranged_search: bool,
    rebuild_trees: bool,
) -> Result<(), String> {
    let all_paths = trees::AllPaths::new(out_dir, data.name());
    if rebuild_trees || !all_paths.all_exist(balanced, permuted, false) {
        trees::build_all(out_dir, data, metric, seed, permuted, balanced, Some(128))?;
    }
    run::<_, _, _, String>(
        &all_paths,
        metric,
        queries,
        None,
        radial_fractions,
        ks,
        max_time,
        run_linear,
        balanced,
        permuted,
        ranged_search,
    )
}

/// Run the workflow of the CAKES benchmarks on a tabular dataset.
pub fn run_tabular<P: AsRef<std::path::Path>, M: ParCountingMetric<Vec<f32>, f32>>(
    out_dir: &P,
    data_name: &str,
    metric: &M,
    queries: &[Vec<f32>],
    num_queries: usize,
    radial_fractions: &[f32],
    ks: &[usize],
    seed: Option<u64>,
    max_time: Duration,
    run_linear: bool,
    balanced: bool,
    permuted: bool,
    ranged_search: bool,
    rebuild_trees: bool,
) -> Result<(), String> {
    let data_path = out_dir.as_ref().join(format!("{data_name}.npy"));
    let data = FlatVec::<Vec<f32>, usize>::read_npy(&data_path)?;

    let neighbors_path = out_dir.as_ref().join(format!("{data_name}-neighbors.npy"));
    let distances_path = out_dir.as_ref().join(format!("{data_name}-distances.npy"));
    let (queries, neighbors) = if neighbors_path.exists() && distances_path.exists() {
        let neighbors = FlatVec::<Vec<u64>, usize>::read_npy(&neighbors_path)?.take_items();
        let neighbors = neighbors
            .into_iter()
            .map(|n| n.into_iter().map(Number::as_usize).collect::<Vec<_>>());

        let distances = FlatVec::<Vec<f32>, usize>::read_npy(&distances_path)?.take_items();

        let neighbors = neighbors
            .zip(distances)
            .map(|(n, d)| n.into_iter().zip(d).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let mut queries = queries.iter().cloned().zip(neighbors).collect::<Vec<_>>();

        let mut rng = rand::rng();
        queries.shuffle(&mut rng);
        let _ = queries.split_off(num_queries);

        let (queries, neighbors): (Vec<_>, Vec<_>) = queries.into_iter().unzip();
        (queries, Some(neighbors))
    } else {
        let mut rng = rand::rng();
        let mut queries = queries.to_vec();
        queries.shuffle(&mut rng);
        let _ = queries.split_off(num_queries);
        (queries, None)
    };
    let neighbors = neighbors.as_deref();

    let all_paths = trees::AllPaths::new(out_dir, data.name());
    if rebuild_trees || !all_paths.all_exist(balanced, permuted, false) {
        trees::build_all(out_dir, &data, metric, seed, permuted, balanced, None)?;
    }
    run::<_, _, _, usize>(
        &all_paths,
        metric,
        &queries,
        neighbors,
        radial_fractions,
        ks,
        max_time,
        run_linear,
        balanced,
        permuted,
        ranged_search,
    )
}

/// Run the full workflow of the CAKES benchmarks on a dataset.
pub fn run<I, T, M, Me>(
    all_paths: &trees::AllPaths,
    metric: &M,
    queries: &[I],
    neighbors: Option<&[Vec<(usize, T)>]>,
    radial_fractions: &[f32],
    ks: &[usize],
    max_time: Duration,
    run_linear: bool,
    balanced: bool,
    permuted: bool,
    ranged_search: bool,
) -> Result<(), String>
where
    I: Send + Sync + Clone + bitcode::Encode + bitcode::Decode,
    T: Number + bitcode::Encode + bitcode::Decode,
    M: ParCountingMetric<I, T>,
    Me: Send + Sync + Clone + bitcode::Encode + bitcode::Decode,
{
    ftlog::info!("Reading Ball from {}...", all_paths.ball.display());
    let ball = Ball::<T>::par_read_from(&all_paths.ball)?;
    let radii = radial_fractions
        .iter()
        .map(|&f| f * ball.radius().as_f32())
        .map(T::from)
        .collect::<Vec<_>>();

    ftlog::info!("Reading data from {}...", all_paths.data.display());
    let data = FlatVec::<I, Me>::par_read_from(&all_paths.data)?;

    let (min_dim, max_dim) = data.dimensionality_hint();
    let mut report = CakesResults::new(
        data.name(),
        data.cardinality(),
        max_dim.unwrap_or(min_dim),
        metric.name(),
    );

    ftlog::info!("Running search algorithms on Ball...");
    search::bench_all_algs(
        &mut report,
        metric,
        queries,
        neighbors,
        &ball,
        &data,
        false,
        false,
        false,
        max_time,
        ks,
        &radii,
        run_linear,
        ranged_search,
    );

    if permuted {
        let ball = PermutedBall::<T, Ball<T>>::par_read_from(&all_paths.permuted_ball)?;
        let data = FlatVec::<I, Me>::par_read_from(&all_paths.permuted_data)?;
        ftlog::info!("Running search algorithms on PermutedBall...");
        search::bench_all_algs(
            &mut report,
            metric,
            queries,
            None,
            &ball,
            &data,
            false,
            true,
            false,
            max_time,
            ks,
            &radii,
            run_linear,
            ranged_search,
        );
    }

    if balanced {
        let ball = Ball::<T>::par_read_from(&all_paths.balanced_ball)?;
        ftlog::info!("Running search algorithms on BalancedBall...");
        search::bench_all_algs(
            &mut report,
            metric,
            queries,
            neighbors,
            &ball,
            &data,
            true,
            false,
            false,
            max_time,
            ks,
            &radii,
            run_linear,
            ranged_search,
        );

        if permuted {
            let ball = PermutedBall::<T, Ball<T>>::par_read_from(&all_paths.permuted_balanced_ball)?;
            let data = FlatVec::<I, Me>::par_read_from(&all_paths.permuted_balanced_data)?;
            ftlog::info!("Running search algorithms on PermutedBalancedBall...");
            search::bench_all_algs(
                &mut report,
                metric,
                queries,
                None,
                &ball,
                &data,
                true,
                true,
                false,
                max_time,
                ks,
                &radii,
                run_linear,
                ranged_search,
            );
        }
    }

    report.write_to_csv(&all_paths.out_dir)
}
