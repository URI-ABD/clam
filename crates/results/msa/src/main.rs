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
    adapter::ParBallAdapter, cakes::OffBall, cluster::WriteCsv, partition::ParPartition, Ball, Cluster, Dataset,
    FlatVec, MetricSpace,
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
    num_samples: Option<usize>,

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

    // let pool = rayon::ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build()
    //     .map_err(|e| e.to_string())?;

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

    let (off_ball, data) = if msa_ball_path.exists() && msa_data_path.exists() {
        ftlog::info!("Reading MSA ball from {msa_ball_path:?}");
        let off_ball = bincode::deserialize_from(std::fs::File::open(&msa_ball_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

        ftlog::info!("Reading MSA data from {msa_data_path:?}");
        let data = bincode::deserialize_from(std::fs::File::open(&msa_data_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

        (off_ball, data)
    } else {
        OffBall::par_from_ball_tree(ball, data)
    };

    let aligner = abd_clam::msa::NeedlemanWunschAligner::<i32>::default();
    let msa_builder = abd_clam::msa::MsaBuilder::new(&aligner).with_binary_tree(&off_ball, &data);
    let aligned_sequences = msa_builder
        .as_msa()
        .into_iter()
        .map(String::from_utf8)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    ftlog::info!("Finished building MSA with {} sequences.", aligned_sequences.len());

    let data = FlatVec::new(aligned_sequences, data.metric().clone())?.with_metadata(data.metadata().to_vec())?;

    let msa_fasta_path = path_manager.msa_fasta_path();
    if !msa_fasta_path.exists() {
        ftlog::info!("Writing MSA to {msa_fasta_path:?}");
        data::write_fasta(&data, &msa_fasta_path)?;
    }

    Ok(())
}
