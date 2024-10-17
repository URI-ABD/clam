#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::style,
    clippy::complexity,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
//! Benchmarks for the CAKES paper.

use std::path::PathBuf;

use abd_clam::{dataset::DatasetIO, Dataset, FlatVec};
use bench_utils::Complex;
use clap::Parser;

mod data_gen;
mod metric;
mod search;
mod trees;
mod workflow;

use distances::Number;
use metric::{CountingMetric, ParCountingMetric};

/// Reproducible results for the CAKES paper.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[allow(clippy::struct_excessive_bools)]
struct Args {
    /// Path to the input file.
    #[arg(short('i'), long)]
    inp_dir: PathBuf,

    /// The dataset to benchmark.
    #[arg(short('d'), long)]
    dataset: bench_utils::RawData,

    /// The number of queries to use for benchmarking.
    #[arg(short('q'), long)]
    num_queries: usize,

    /// Whether to count the number of distance computations during search.
    #[arg(short('c'), long, default_value = "false")]
    count_distance_calls: bool,

    /// This parameter is used differently depending on the dataset:
    ///
    /// - For any vector datasets, this is the maximum power of 2 to which the
    ///   cardinality should be augmented for scaling experiments.
    /// - For 'omic datasets, this is the maximum power of 2 by which the
    ///   cardinality should be divided (sub-sampled) for scaling experiments.
    /// - For the complex-valued radio-ml dataset, this works identically as
    ///   with the sequence datasets.
    /// - For set datasets (kosarak, etc.), this is ignored.
    #[arg(short('m'), long)]
    max_power: Option<u32>,

    /// The minimum power of 2 to which the cardinality of the dataset should be
    /// augmented for scaling experiments.
    ///
    /// This is only used with the tabular floating-point datasets and is
    /// ignored otherwise.
    #[arg(short('n'), long, default_value = "0")]
    min_power: Option<u32>,

    /// The seed for the random number generator.
    #[arg(short('s'), long)]
    seed: Option<u64>,

    /// The maximum time, in seconds, to run each algorithm.
    #[arg(short('t'), long, default_value = "10.0")]
    max_time: f32,

    /// Whether to run benchmarks with balanced trees.
    #[arg(short('b'), long)]
    balanced_trees: bool,

    /// Whether to run benchmarks with permuted data.
    #[arg(short('p'), long)]
    permuted_data: bool,

    /// Whether to run ranged search benchmarks.
    #[arg(short('r'), long)]
    ranged_search: bool,

    /// Path to the output directory.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,

    /// Stop after generating the augmented datasets.
    #[arg(short('g'), long)]
    generate_only: bool,

    /// Whether to run linear search on the datasets to find the ground truth.
    #[arg(short('l'), long)]
    linear_search: bool,

    /// Whether to rebuild the trees.
    #[arg(short('w'), long)]
    rebuild_trees: bool,
}

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
fn main() -> Result<(), String> {
    let args = Args::parse();
    println!("Args: {args:?}");

    let log_name = format!("cakes-{}", args.dataset.name());
    let (_guard, log_path) = bench_utils::configure_logger(&log_name)?;
    println!("Log file: {log_path:?}");

    ftlog::info!("{args:?}");

    // Check the input and output directories.
    let inp_dir = args.inp_dir.canonicalize().map_err(|e| e.to_string())?;
    ftlog::info!("Input directory: {inp_dir:?}");

    let out_dir = if let Some(out_dir) = args.out_dir {
        out_dir
    } else {
        ftlog::info!("No output directory specified. Using default.");
        let mut out_dir = inp_dir
            .parent()
            .ok_or("No parent directory of `inp_dir`")?
            .to_path_buf();
        out_dir.push(format!("{}_results", args.dataset.name()));
        if !out_dir.exists() {
            std::fs::create_dir(&out_dir).map_err(|e| e.to_string())?;
        }
        out_dir
    }
    .canonicalize()
    .map_err(|e| e.to_string())?;
    ftlog::info!("Output directory: {out_dir:?}");

    let radial_fractions = [0.1, 0.25];
    let ks = [10, 100];
    let seed = args.seed;
    let min_power = args.min_power.unwrap_or_default();
    let max_power = args.max_power.unwrap_or(5);
    let max_time = std::time::Duration::from_secs_f32(args.max_time);

    if min_power > max_power {
        return Err("min_power must be less than or equal to max_power".to_string());
    }

    if args.dataset.is_tabular() {
        let metric = {
            let mut metric: Box<dyn ParCountingMetric<_, _>> = match args.dataset.metric() {
                "cosine" => Box::new(metric::Cosine::new()),
                "euclidean" => Box::new(metric::Euclidean::new()),
                _ => return Err(format!("Unknown metric: {}", args.dataset.metric())),
            };
            if !args.count_distance_calls {
                metric.disable_counting();
            }
            metric
        };

        let gt_metric = if args.linear_search { Some(&metric) } else { None };
        let queries = if matches!(args.dataset, bench_utils::RawData::Random) {
            data_gen::read_or_gen_random(gt_metric, max_power, seed, &out_dir)?
        } else {
            data_gen::read_tabular_and_augment(&args.dataset, gt_metric, max_power, args.seed, &inp_dir, &out_dir)?
        };
        if args.generate_only {
            return Ok(());
        }

        for power in min_power..=max_power {
            let data_name = format!("{}-{power}", args.dataset.name());
            ftlog::info!("Reading {}x augmented data...", 1 << power);
            let run_linear = power < 4;

            workflow::run_tabular(
                &out_dir,
                &data_name,
                &metric,
                &queries,
                args.num_queries,
                &radial_fractions,
                &ks,
                seed,
                max_time,
                run_linear,
                args.balanced_trees,
                args.permuted_data,
                args.ranged_search,
                args.rebuild_trees,
            )?;
        }
    } else if args.dataset.is_sequence() {
        let metric = {
            let mut metric: Box<dyn ParCountingMetric<String, u32>> = match args.dataset.metric() {
                "levenshtein" => Box::new(metric::Levenshtein::new()),
                "hamming" => Box::new(metric::Hamming::new()),
                _ => return Err(format!("Unknown metric: {}", args.dataset.metric())),
            };
            if !args.count_distance_calls {
                metric.disable_counting();
            }
            metric
        };

        let (queries, subsampled_paths) =
            data_gen::read_fasta_and_subsample(&inp_dir, &out_dir, false, args.num_queries, max_power, seed)?;
        let queries = queries.into_iter().map(|(_, q)| q).collect::<Vec<_>>();
        if args.generate_only {
            return Ok(());
        }

        ftlog::info!("Found {} sub-sampled datasets:", subsampled_paths.len());
        for p in &subsampled_paths {
            ftlog::info!("{p:?}");
        }

        for (i, sample_path) in subsampled_paths.iter().enumerate() {
            if i.as_u32() < min_power {
                continue;
            }

            ftlog::info!("Reading sub-sampled data from {sample_path:?}...");
            let data = FlatVec::<String, String>::read_from(sample_path)?;
            ftlog::info!("Data from {sample_path:?} has {} sequences...", data.cardinality());

            let run_linear = i < 4;

            ftlog::info!("Running workflow for {}...", data.name());
            workflow::run_fasta(
                &out_dir,
                &data,
                &metric,
                &queries,
                &radial_fractions,
                &ks,
                seed,
                max_time,
                run_linear,
                args.balanced_trees,
                args.permuted_data,
                args.ranged_search,
                args.rebuild_trees,
            )?;
        }
    } else if matches!(args.dataset, bench_utils::RawData::RadioML) {
        let metric = {
            let mut metric = metric::DynamicTimeWarping::new();
            if !args.count_distance_calls {
                <metric::DynamicTimeWarping as CountingMetric<Vec<Complex<f64>>, f64>>::disable_counting(&mut metric);
            }
            metric
        };

        let snr = Some(10);
        let (queries, subsampled_paths) =
            data_gen::read_radio_ml_and_subsample(&inp_dir, &out_dir, args.num_queries, max_power, seed, snr)?;
        if args.generate_only {
            return Ok(());
        }

        ftlog::info!("Found {} sub-sampled datasets:", subsampled_paths.len());
        for p in &subsampled_paths {
            ftlog::info!("{p:?}");
        }

        for (i, sample_path) in subsampled_paths.iter().enumerate() {
            if i.as_u32() < min_power {
                continue;
            }

            ftlog::info!("Reading sub-sampled data from {sample_path:?}...");
            let data = FlatVec::<Vec<Complex<f64>>, usize>::read_from(sample_path)?;
            ftlog::info!("Data from {sample_path:?} has {} signals...", data.cardinality());

            let run_linear = i < 3;

            ftlog::info!("Running workflow for {}...", data.name());
            workflow::run_radio_ml(
                &out_dir,
                &data,
                &metric,
                &queries,
                &radial_fractions,
                &ks,
                seed,
                max_time,
                run_linear,
                args.balanced_trees,
                args.permuted_data,
                args.ranged_search,
                args.rebuild_trees,
            )?;
        }
    } else {
        let msg = format!("Unsupported dataset: {}", args.dataset.name());
        ftlog::error!("{msg}");
        return Err(msg);
    }

    Ok(())
}
