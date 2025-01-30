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
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
#![doc = include_str!("../README.md")]

use std::path::PathBuf;

use abd_clam::{
    cakes::PermutedBall,
    cluster::{adapter::ParBallAdapter, ClusterIO},
    dataset::{DatasetIO, ParDatasetIO},
    pancakes::{CodecData, SquishyBall},
    Ball, Cluster, Dataset, FlatVec,
};
use bench_cakes::metric::CountingMetric;
use bench_utils::reports::CakesResults;
use clap::Parser;
use data::MembershipSet;

mod data;

/// Reproducible results for the PANCAKES paper.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[allow(clippy::struct_excessive_bools)]
struct Args {
    /// Path to the input file.
    #[arg(short('i'), long)]
    inp_dir: PathBuf,

    /// Path to the output directory.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,

    /// The dataset to benchmark.
    #[arg(short('d'), long)]
    dataset: bench_utils::RawData,

    /// The number of queries to use for benchmarking.
    #[arg(short('q'), long)]
    num_queries: usize,

    /// The seed for the random number generator.
    #[arg(short('s'), long)]
    seed: Option<u64>,

    /// The maximum time, in seconds, to run each algorithm.
    #[arg(short('t'), long, default_value = "10.0")]
    max_time: f32,

    /// Whether to run ranged search benchmarks.
    #[arg(short('r'), long)]
    ranged_search: bool,

    /// Whether to rebuild the trees.
    #[arg(short('w'), long)]
    rebuild_trees: bool,

    /// Whether to run the search benchmarks only for the compressed data.
    #[arg(short('c'), long)]
    compressed_only: bool,
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), String> {
    let args = Args::parse();
    println!("Args: {args:?}");

    let log_name = format!("pancakes-{}", args.dataset.name());
    let (_guard, log_path) = bench_utils::configure_logger(&log_name)?;
    println!("Log file: {log_path:?}");

    ftlog::info!("{args:?}");

    // Check the input and output directories.
    let inp_dir = args.inp_dir.canonicalize().map_err(|e| e.to_string())?;
    if !inp_dir.exists() {
        return Err(format!("{inp_dir:?} does not exist."));
    }
    if !inp_dir.is_dir() {
        return Err(format!("{inp_dir:?} is not a directory."));
    }
    ftlog::info!("Input directory: {inp_dir:?}");

    let out_dir = if let Some(out_dir) = args.out_dir {
        if !out_dir.exists() {
            ftlog::info!("Creating output directory: {out_dir:?}");
            std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
        }
        if !out_dir.is_dir() {
            return Err(format!("{out_dir:?} is not a directory."));
        }
        out_dir
    } else {
        ftlog::info!("No output directory specified, using the parent of the input directory.");
        inp_dir.parent().ok_or("No parent directory.")?.to_path_buf()
    };
    ftlog::info!("Output directory: {out_dir:?}");

    let data_name = args.dataset.name();
    let radial_fractions = [0.1_f32, 0.25];
    let ks = [10, 100];
    let max_time = std::time::Duration::from_secs_f32(args.max_time);

    if args.dataset.is_set() {
        let metric = {
            let mut metric = bench_cakes::metric::Jaccard::default();
            <bench_cakes::metric::Jaccard as CountingMetric<MembershipSet, f32>>::disable_counting(&mut metric);
            metric
        };

        let (data, queries) = bench_cakes::data::sets::read(&args.inp_dir, data_name, args.num_queries, args.seed).map(
            |(data, queries)| {
                (
                    data.transform_items(data::MembershipSet::from),
                    queries.into_iter().map(data::MembershipSet::from).collect::<Vec<_>>(),
                )
            },
        )?;

        let all_paths = bench_cakes::workflow::trees::AllPaths::new(&out_dir, data_name);
        if args.rebuild_trees || !all_paths.all_exist(false, true, false) {
            bench_cakes::workflow::trees::build_all(&out_dir, &data, &metric, args.seed, true, false, None)?;
        }

        // Create the `Ball` tree and set up radii for ranged search.
        ftlog::info!("Reading the Ball tree.");
        let ball = Ball::read_from(&all_paths.ball)?;
        let radii = radial_fractions.iter().map(|&f| f * ball.radius()).collect::<Vec<_>>();

        // Create the `PermutedBall` tree and its associated permuted data.
        ftlog::info!("Reading the PermutedBall tree.");
        let perm_ball = PermutedBall::<f32, Ball<f32>>::read_from(&all_paths.permuted_ball)?;
        let perm_data = FlatVec::<MembershipSet, usize>::read_from(&all_paths.permuted_data)?;
        let perm_data = perm_data.with_name(&format!("{}-permuted", args.dataset.name()));

        // Run the Search benchmarks before compression.
        let (min_dim, max_dim) = perm_data.dimensionality_hint();
        if !args.compressed_only {
            ftlog::info!("Running the search benchmarks before compression.");
            let mut report = CakesResults::<f32>::new(
                perm_data.name(),
                perm_data.cardinality(),
                max_dim.unwrap_or(min_dim),
                "jaccard",
            );
            bench_cakes::workflow::search::bench_all_algs(
                &mut report,
                &metric,
                &queries,
                None,
                &perm_ball,
                &perm_data,
                false,
                true,
                max_time,
                &ks,
                &radii,
                true,
                true,
            );
            report.write_to_csv(&out_dir)?;
        }

        let (squishy_ball, codec_data) =
            if !args.rebuild_trees && all_paths.squishy_ball.exists() && all_paths.codec_data.exists() {
                // Load the `SquishyBall` tree and `CodecData` from disk.
                ftlog::info!("SquishyBall and CodecData already exist, loading from disk.");
                (
                    SquishyBall::read_from(&all_paths.squishy_ball)?,
                    CodecData::par_read_from(&all_paths.codec_data)?,
                )
            } else {
                // Create the `SquishyBall` tree and `CodecData`.
                ftlog::info!("Building the SquishyBall tree.");
                let (squishy_ball, codec_data) = SquishyBall::par_from_ball_tree(ball, data, &metric);
                let codec_data = codec_data.with_name(&format!("{}-codec", args.dataset.name()));

                // Save the squishy ball and codec data to disk.
                squishy_ball.write_to(&all_paths.squishy_ball)?;
                codec_data.par_write_to(&all_paths.codec_data)?;

                (squishy_ball, codec_data)
            };

        // Run the Search benchmarks after compression.
        ftlog::info!("Running the search benchmarks after compression.");
        let mut report = CakesResults::<f32>::new(
            codec_data.name(),
            codec_data.cardinality(),
            max_dim.unwrap_or(min_dim),
            "jaccard",
        );
        bench_cakes::workflow::search::bench_all_algs(
            &mut report,
            &metric,
            &queries,
            None,
            &squishy_ball,
            &codec_data,
            false,
            true,
            max_time,
            &ks,
            &radii,
            true,
            true,
        );
        report.write_to_csv(&out_dir)?;
    }

    Ok(())
}
