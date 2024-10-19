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
    adapter::{ParAdapter, ParBallAdapter},
    cakes::OffBall,
    cluster::WriteCsv,
    msa::PartialMSA,
    partition::ParPartition,
    Ball, Cluster, Dataset, FlatVec, Metric,
};
use clap::Parser;

use results_cakes::{data::PathManager, utils::configure_logger};

mod data;

/// Reproducible results for the CAKES and panCAKES papers.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset type
    #[arg(short('d'), long)]
    dataset: data::RawData,

    /// The number of samples to use for the dataset.
    #[arg(short('n'), long)]
    num_samples: usize,

    /// Path to the input file.
    #[arg(short('i'), long)]
    inp_path: PathBuf,

    /// Path to the output directory.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,
}

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
fn main() -> Result<(), String> {
    let args = Args::parse();

    let log_name = format!("msa-{}", args.dataset.name());
    let (_guard, log_path) = configure_logger(&log_name)?;
    println!("Log file: {log_path:?}");

    ftlog::info!("{args:?}");

    let inp_path = args.inp_path.canonicalize().map_err(|e| e.to_string())?;

    let out_dir = if let Some(out_dir) = args.out_dir {
        out_dir
    } else {
        ftlog::info!("No output directory specified. Using default.");
        let mut out_dir = inp_path
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

    ftlog::info!("Input file: {inp_path:?}");
    ftlog::info!("Output directory: {out_dir:?}");

    let data = args.dataset.read(&inp_path, &out_dir, args.num_samples)?;
    ftlog::info!("Finished reading dataset.");
    let path_manager = PathManager::new(data.name(), &out_dir);

    let ball_path = path_manager.ball_path();
    ftlog::info!("Ball path: {ball_path:?}");
    let ball = if ball_path.exists() {
        ftlog::info!("Reading ball from {ball_path:?}");
        // Deserialize the ball from disk.
        bincode::deserialize_from(std::fs::File::open(&ball_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
    } else {
        // Create the ball from scratch.
        ftlog::info!("Building ball.");
        let mut depth = 0;
        let seed = Some(42);

        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let mut ball = Ball::par_new(&data, &indices, 0, seed);
        let depth_delta = ball.max_recursion_depth();

        let criteria = |c: &Ball<_, _, _>| c.depth() < 1;
        ball.par_partition(&data, &criteria, seed);

        while ball.leaves().into_iter().any(|c| !c.is_singleton()) {
            depth += depth_delta;
            let criteria = |c: &Ball<_, _, _>| c.depth() < depth;
            ball.par_partition_further(&data, &criteria, seed);
        }

        let num_leaves = ball.leaves().len();
        ftlog::info!("Built ball with {num_leaves} leaves.");

        // Serialize the ball to disk.
        ftlog::info!("Writing ball to {ball_path:?}");
        bincode::serialize_into(std::fs::File::create(&ball_path).map_err(|e| e.to_string())?, &ball)
            .map_err(|e| e.to_string())?;

        // Write the ball to a CSV file.
        let csv_path = path_manager.ball_csv_path();
        ftlog::info!("Writing ball to CSV at {csv_path:?}");
        ball.write_to_csv(&csv_path)?;

        ball
    };

    ftlog::info!("Finished building/reading ball with {} leaves.", ball.leaves().len());

    let msa_ball_path = path_manager.msa_ball_path();
    let msa_data_path = path_manager.msa_data_path();

    let (msa_root, data) = if msa_ball_path.exists() && msa_data_path.exists() {
        ftlog::info!("Reading MSA ball from {msa_ball_path:?}");
        let msa_ball = bincode::deserialize_from(std::fs::File::open(&msa_ball_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

        ftlog::info!("Reading MSA data from {msa_data_path:?}");
        let msa_data = bincode::deserialize_from(std::fs::File::open(&msa_data_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

        (msa_ball, msa_data)
    } else {
        let (off_ball, data) = OffBall::par_from_ball_tree(ball, data);
        let msa_root = PartialMSA::par_adapt_tree_iterative(off_ball, None, &data);
        ftlog::info!("Finished building MSA tree.");

        let aligned_sequences = msa_root.full_msa(&data);
        let width = aligned_sequences[0].len();

        let distance_fn = |x: &String, y: &String| distances::strings::hamming::<u32>(x, y);
        let msa_metric = Metric::new(distance_fn, false);
        let aligned_data = FlatVec::new(aligned_sequences, msa_metric)?
            .with_dim_lower_bound(width)
            .with_dim_upper_bound(width)
            .with_metadata(data.metadata().to_vec())?;

        ftlog::info!("Writing MSA ball to {msa_ball_path:?}");
        bincode::serialize_into(
            std::fs::File::create(&msa_ball_path).map_err(|e| e.to_string())?,
            &msa_root,
        )
        .map_err(|e| e.to_string())?;

        ftlog::info!("Writing MSA to {msa_data_path:?}");
        bincode::serialize_into(
            std::fs::File::create(&msa_data_path).map_err(|e| e.to_string())?,
            &aligned_data,
        )
        .map_err(|e| e.to_string())?;

        (msa_root, aligned_data)
    };

    ftlog::info!("Finished building MSA with {} sequences.", msa_root.cardinality());

    let msa_fasta_path = path_manager.msa_fasta_path();
    if !msa_fasta_path.exists() {
        ftlog::info!("Writing MSA to {msa_fasta_path:?}");
        data::write_fasta(&data, &msa_fasta_path)?;
    }

    Ok(())
}
