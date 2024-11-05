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

use core::ops::Neg;
use std::path::PathBuf;

use abd_clam::{
    adapter::ParBallAdapter,
    cakes::OffBall,
    cluster::WriteCsv,
    msa::{self, Aligner, CostMatrix, Msa},
    partition::ParPartition,
    Ball, Cluster, Dataset, FlatVec, Metric,
};
use clap::Parser;

use distances::Number;
use results_cakes::{data::PathManager, utils::configure_logger};

mod data;

/// Reproducible results for the MSA paper.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the input fasta file.
    #[arg(short('i'), long)]
    inp_path: PathBuf,

    /// The number of samples to use for the dataset.
    #[arg(short('n'), long)]
    num_samples: Option<usize>,

    /// The cost matrix to use for the alignment.
    #[arg(short('m'), long)]
    cost_matrix: SpecialMatrix,

    /// Path to the output directory.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,
}

/// The datasets we use for benchmarks.
#[derive(clap::ValueEnum, Debug, Clone)]
#[allow(non_camel_case_types, clippy::doc_markdown)]
#[non_exhaustive]
pub enum SpecialMatrix {
    /// The default matrix.
    #[clap(name = "default")]
    Default,
    /// Default but with affine gap penalties. Gap opening is 10 and ext is 1.
    #[clap(name = "default-affine")]
    DefaultAffine,
    /// Extended IUPAC matrix.
    #[clap(name = "extended-iupac")]
    ExtendedIupac,
    /// Blosum62 matrix.
    #[clap(name = "blosum62")]
    Blosum62,
}

impl SpecialMatrix {
    /// Get the cost matrix.
    #[must_use]
    pub fn cost_matrix<T: Number + Neg<Output = T>>(&self) -> CostMatrix<T> {
        match self {
            Self::Default => CostMatrix::default(),
            Self::DefaultAffine => CostMatrix::default_affine(),
            Self::ExtendedIupac => CostMatrix::extended_iupac(),
            Self::Blosum62 => CostMatrix::blosum62(),
        }
    }

    /// Whether the matrix is used for minimization.
    #[must_use]
    pub const fn is_minimizer(&self) -> bool {
        match self {
            Self::Default | Self::DefaultAffine | Self::ExtendedIupac => true,
            Self::Blosum62 => false,
        }
    }
}

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
fn main() -> Result<(), String> {
    let args = Args::parse();
    ftlog::info!("{args:?}");

    // let pool = rayon::ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build()
    //     .map_err(|e| e.to_string())?;

    let fasta_file = data::FastaFile::new(args.inp_path, args.out_dir)?;

    let log_name = format!("msa-{}", fasta_file.name());
    let (_guard, log_path) = configure_logger(&log_name)?;
    println!("Log file: {log_path:?}");

    ftlog::info!("Input file: {:?}", fasta_file.raw_path());
    ftlog::info!("Output directory: {:?}", fasta_file.out_dir());

    let data = fasta_file.read(args.num_samples)?;
    ftlog::info!("Finished reading dataset.");
    let path_manager = PathManager::new(data.name(), fasta_file.out_dir());

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
        let (off_ball, data) = OffBall::par_from_ball_tree(ball, data);

        ftlog::info!("Writing MSA ball to {msa_ball_path:?}");
        bincode::serialize_into(
            std::fs::File::create(&msa_ball_path).map_err(|e| e.to_string())?,
            &off_ball,
        )
        .map_err(|e| e.to_string())?;

        ftlog::info!("Writing MSA data to {msa_data_path:?}");
        bincode::serialize_into(std::fs::File::create(&msa_data_path).map_err(|e| e.to_string())?, &data)
            .map_err(|e| e.to_string())?;

        (off_ball, data)
    };

    ftlog::info!(
        "Finished adapting/reading Offset Ball with {} leaves.",
        off_ball.leaves().len()
    );

    let msa_fasta_path = path_manager.msa_fasta_path();

    let hamming_fn = |x: &String, y: &String| distances::strings::hamming::<u32>(x, y);
    let metric = Metric::new(hamming_fn, false);

    let data = if msa_fasta_path.exists() {
        let ([aligned_sequences, _], [width, _]) = results_cakes::data::fasta::read(&msa_fasta_path, 0)?;
        let (aligned_sequences, metadata): (Vec<_>, Vec<_>) = aligned_sequences.into_iter().unzip();
        FlatVec::new(aligned_sequences, metric)?
            .with_metadata(&metadata)?
            .with_dim_lower_bound(width)
            .with_dim_upper_bound(width)
    } else {
        ftlog::info!("Setting up aligner...");
        let cost_matrix = args.cost_matrix.cost_matrix::<i32>();
        let aligner = if args.cost_matrix.is_minimizer() {
            Aligner::new_minimizer(&cost_matrix, b'-')
        } else {
            Aligner::new_maximizer(&cost_matrix, b'-')
        };

        ftlog::info!("Aligning sequences...");
        let builder = msa::Builder::new(&aligner).par_with_binary_tree(&off_ball, &data);

        ftlog::info!("Extracting aligned sequences...");
        let msa = Msa::par_from_builder(&builder);
        let aligned_sequences = msa.strings();
        let width = builder.width();

        ftlog::info!("Finished aligning {} sequences.", builder.len());
        let data = FlatVec::new(aligned_sequences, metric)?.with_metadata(data.metadata())?;

        ftlog::info!("Writing MSA to {msa_fasta_path:?}");
        data::write_fasta(&data, &msa_fasta_path)?;

        data.with_dim_lower_bound(width).with_dim_upper_bound(width)
    };

    ftlog::info!(
        "Finished reading/aligning {} sequences with width = {}.",
        data.cardinality(),
        data.dimensionality_hint().0
    );

    let ps_metric = data.par_scoring_pairwise(b'-', 1, 1);
    // let ps_metric = data.par_scoring_pairwise_subsample(b'-', 1, 1);
    ftlog::info!("Pairwise scoring metric: {ps_metric}");

    Ok(())
}
