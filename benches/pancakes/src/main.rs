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
    adapters::ParBallAdapter,
    cakes::PermutedBall,
    dataset::{AssociatesMetadata, AssociatesMetadataMut},
    pancakes::{CodecData, SquishyBall},
    Ball, Cluster, Dataset, FlatVec, ParDiskIO,
};
use bench_utils::reports::CakesResults;
use clap::Parser;
use data::{MembershipSet, Sequence};
use distances::Number;

mod data;
mod metrics;

// TODO: Add argument for the aligner to use for -omic data.

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

    /// The power of 2 by which the cardinality of the full dataset is reduced.
    #[arg(short('m'), long)]
    max_power: Option<u32>,

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

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
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
        let metric = metrics::Jaccard;
        let (encoder, decoder) = (metric.clone(), metric.clone());

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
        let ball = Ball::par_read_from(&all_paths.ball)?;
        let radii = radial_fractions.iter().map(|&f| f * ball.radius()).collect::<Vec<_>>();

        // Create the `PermutedBall` tree and its associated permuted data.
        ftlog::info!("Reading the PermutedBall tree.");
        let perm_ball = PermutedBall::<f32, Ball<f32>>::par_read_from(&all_paths.permuted_ball)?;
        let perm_data = FlatVec::<MembershipSet, usize>::par_read_from(&all_paths.permuted_data)?;
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
                ftlog::info!("SquishyBall already exists, loading from disk.");
                let squishy_ball = SquishyBall::par_read_from(&all_paths.squishy_ball)?;
                ftlog::info!("CodecData already exists, loading from disk.");
                let codec_data = CodecData::par_read_from(&all_paths.codec_data)?;
                (squishy_ball, codec_data)
            } else {
                // Create the `SquishyBall` tree and `CodecData`.
                ftlog::info!("Building the SquishyBall tree.");
                let (squishy_ball, data) = SquishyBall::par_from_ball_tree(ball, data, &metric);
                let codec_data = CodecData::par_from_compressible(&data, &squishy_ball, encoder, decoder);
                let codec_data = codec_data
                    .with_name(&format!("{}-codec", args.dataset.name()))
                    .with_metadata(data.metadata())?;

                // Save the squishy ball and codec data to disk.
                squishy_ball.par_write_to(&all_paths.squishy_ball)?;
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
            false,
            true,
            max_time,
            &ks,
            &radii,
            true,
            true,
        );
        report.write_to_csv(&out_dir)?;
    } else if args.dataset.is_aligned_sequence() {
        let metric = metrics::Hamming;
        let encoder = metric.clone();
        let decoder = metric.clone();

        let (queries, data_paths) = bench_cakes::data::fasta::read_and_subsample(
            &args.inp_dir,
            &out_dir,
            false,
            args.num_queries,
            args.max_power.unwrap_or(0),
            args.seed,
        )?;

        let data = FlatVec::<String, String>::par_read_from(&data_paths[0])?;
        let data = data.transform_items(Sequence::from);
        let queries = queries.into_iter().map(|(_, q)| Sequence::from(q)).collect::<Vec<_>>();

        let all_paths = bench_cakes::workflow::trees::AllPaths::new(&out_dir, data.name());
        if args.rebuild_trees || !all_paths.all_exist(false, true, false) {
            bench_cakes::workflow::trees::build_all::<_, _, u32, _, _>(
                &out_dir, &data, &metric, args.seed, true, false, None,
            )?;
        }

        // Create the `Ball` tree and set up radii for ranged search.
        ftlog::info!("Reading the Ball tree.");
        let ball = Ball::<u32>::par_read_from(&all_paths.ball)?;
        let radii = radial_fractions
            .iter()
            .map(|&f| f * ball.radius().as_f32())
            .map(Number::as_u32)
            .collect::<Vec<_>>();

        // Create the `PermutedBall` tree and its associated permuted data.
        ftlog::info!("Reading the PermutedBall tree.");
        let perm_ball = PermutedBall::<u32, Ball<u32>>::par_read_from(&all_paths.permuted_ball)?;
        let perm_data = FlatVec::<String, String>::par_read_from(&all_paths.permuted_data)?;
        let perm_data = perm_data.transform_items(Sequence::from);
        let perm_data = perm_data.with_name(&format!("{}-permuted", args.dataset.name()));

        // Run the Search benchmarks before compression.
        let (min_dim, max_dim) = perm_data.dimensionality_hint();
        if !args.compressed_only {
            ftlog::info!("Running the search benchmarks before compression.");
            let mut report = CakesResults::<u32>::new(
                perm_data.name(),
                perm_data.cardinality(),
                max_dim.unwrap_or(min_dim),
                "hamming",
            );
            bench_cakes::workflow::search::bench_all_algs(
                &mut report,
                &metric,
                &queries,
                None,
                &perm_ball,
                &perm_data,
                false,
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
                    SquishyBall::par_read_from(&all_paths.squishy_ball)?,
                    CodecData::<Sequence, String, _, _>::par_read_from(&all_paths.codec_data)?,
                )
            } else {
                // Create the `SquishyBall` tree and `CodecData`.
                ftlog::info!("Building the SquishyBall tree.");
                let (squishy_ball, data) = SquishyBall::<u32, Ball<u32>>::par_from_ball_tree(ball, data, &metric);
                let codec_data = CodecData::par_from_compressible(&data, &squishy_ball, encoder, decoder);
                let codec_data = codec_data
                    .with_name(&format!("{}-codec", args.dataset.name()))
                    .with_metadata(data.metadata())?;

                // Save the squishy ball and codec data to disk.
                squishy_ball.par_write_to(&all_paths.squishy_ball)?;
                codec_data.par_write_to(&all_paths.codec_data)?;

                (squishy_ball, codec_data)
            };

        // Run the Search benchmarks after compression.
        ftlog::info!("Running the search benchmarks after compression.");
        let mut report = CakesResults::<u32>::new(
            codec_data.name(),
            codec_data.cardinality(),
            max_dim.unwrap_or(min_dim),
            "hamming",
        );
        bench_cakes::workflow::search::bench_all_algs(
            &mut report,
            &metric,
            &queries,
            None,
            &squishy_ball,
            &codec_data,
            false,
            false,
            true,
            max_time,
            &ks,
            &radii,
            true,
            true,
        );
        report.write_to_csv(&out_dir)?;
    } else if args.dataset.is_unaligned_sequence() {
        let metric = metrics::Levenshtein;
        let encoder = args.dataset.aligner::<u32>();
        let decoder = args.dataset.aligner::<u32>();

        let (queries, data_paths) =
            bench_cakes::data::fasta::read_and_subsample(&args.inp_dir, &out_dir, true, args.num_queries, 0, args.seed)?;

        let data = FlatVec::<String, String>::par_read_from(&data_paths[0])?;
        let data = data.transform_items(Sequence::from);
        let queries = queries.into_iter().map(|(_, q)| Sequence::from(q)).collect::<Vec<_>>();

        let all_paths = bench_cakes::workflow::trees::AllPaths::new(&out_dir, data.name());
        if args.rebuild_trees || !all_paths.all_exist(false, true, false) {
            bench_cakes::workflow::trees::build_all::<_, _, u32, _, _>(
                &out_dir, &data, &metric, args.seed, true, false, None,
            )?;
        }

        // Create the `Ball` tree and set up radii for ranged search.
        ftlog::info!("Reading the Ball tree.");
        let ball = Ball::<u32>::par_read_from(&all_paths.ball)?;
        let radii = radial_fractions
            .iter()
            .map(|&f| f * ball.radius().as_f32())
            .map(Number::as_u32)
            .collect::<Vec<_>>();

        // Create the `PermutedBall` tree and its associated permuted data.
        ftlog::info!("Reading the PermutedBall tree.");
        let perm_ball = PermutedBall::<u32, Ball<u32>>::par_read_from(&all_paths.permuted_ball)?;
        let perm_data = FlatVec::<String, String>::par_read_from(&all_paths.permuted_data)?;
        let perm_data = perm_data.transform_items(Sequence::from);
        let perm_data = perm_data.with_name(&format!("{}-permuted", args.dataset.name()));

        // Run the Search benchmarks before compression.
        let (min_dim, max_dim) = perm_data.dimensionality_hint();
        if !args.compressed_only {
            ftlog::info!("Running the search benchmarks before compression.");
            let mut report = CakesResults::<u32>::new(
                perm_data.name(),
                perm_data.cardinality(),
                max_dim.unwrap_or(min_dim),
                "levenshtein",
            );
            bench_cakes::workflow::search::bench_all_algs(
                &mut report,
                &metric,
                &queries,
                None,
                &perm_ball,
                &perm_data,
                false,
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
                    SquishyBall::par_read_from(&all_paths.squishy_ball)?,
                    CodecData::<Sequence, String, _, _>::par_read_from(&all_paths.codec_data)?,
                )
            } else {
                // Create the `SquishyBall` tree and `CodecData`.
                ftlog::info!("Building the SquishyBall tree.");
                let (squishy_ball, data) = SquishyBall::<u32, Ball<u32>>::par_from_ball_tree(ball, data, &metric);
                let codec_data = CodecData::par_from_compressible(&data, &squishy_ball, encoder, decoder);
                let codec_data = codec_data
                    .with_name(&format!("{}-codec", args.dataset.name()))
                    .with_metadata(data.metadata())?;

                // Save the squishy ball and codec data to disk.
                squishy_ball.par_write_to(&all_paths.squishy_ball)?;
                codec_data.par_write_to(&all_paths.codec_data)?;

                (squishy_ball, codec_data)
            };

        // Run the Search benchmarks after compression.
        ftlog::info!("Running the search benchmarks after compression.");
        let mut report = CakesResults::<u32>::new(
            codec_data.name(),
            codec_data.cardinality(),
            max_dim.unwrap_or(min_dim),
            "levenshtein",
        );
        bench_cakes::workflow::search::bench_all_algs(
            &mut report,
            &metric,
            &queries,
            None,
            &squishy_ball,
            &codec_data,
            false,
            false,
            true,
            max_time,
            &ks,
            &radii,
            true,
            true,
        );
        report.write_to_csv(&out_dir)?;
    } else {
        unimplemented!("Dataset {data_name} is not supported.");
    }

    Ok(())
}
