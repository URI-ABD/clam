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

use abd_clam::{msa::CostMatrix, Cluster, Dataset, Metric, MetricSpace};
use clap::Parser;

use distances::Number;
use results_cakes::{data::PathManager, utils::configure_logger};

mod data;
mod steps;

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

fn main() -> Result<(), String> {
    let args = Args::parse();
    ftlog::info!("{args:?}");

    let fasta_file = data::FastaFile::new(args.inp_path, args.out_dir)?;

    let log_name = format!("msa-{}", fasta_file.name());
    // We need the `_guard` in scope to ensure proper logging.
    let (_guard, log_path) = configure_logger(&log_name)?;
    println!("Log file: {log_path:?}");

    ftlog::info!("Input file: {:?}", fasta_file.raw_path());
    ftlog::info!("Output directory: {:?}", fasta_file.out_dir());

    let data = fasta_file.read(args.num_samples)?;
    ftlog::info!("Finished reading original dataset.");
    let path_manager = PathManager::new(data.name(), fasta_file.out_dir());

    // Set up the Hamming metric for the aligned sequences.
    let msa_fasta_path = path_manager.msa_fasta_path();
    let hamming_fn = |x: &String, y: &String| distances::strings::hamming::<u32>(x, y);
    let hamming_metric = Metric::new(hamming_fn, false);

    let data = if msa_fasta_path.exists() {
        // Read the aligned sequences.
        steps::read_aligned(msa_fasta_path, hamming_metric)?
    } else {
        let msa_ball_path = path_manager.msa_ball_path();
        let msa_data_path = path_manager.msa_data_path();

        let (off_ball, data) = if msa_ball_path.exists() && msa_data_path.exists() {
            // Read the Offset Ball and the dataset.
            steps::read_offset_ball(msa_ball_path, msa_data_path, data.metric().clone())?
        } else {
            let ball_path = path_manager.ball_path();
            let ball = if ball_path.exists() {
                // Read the Ball.
                steps::read_ball(ball_path)?
            } else {
                // Build the Ball.
                steps::build_ball(&data, ball_path, path_manager.ball_csv_path())?
            };
            ftlog::info!("Ball has {} leaves.", ball.leaves().len());

            // Build the Offset Ball and the dataset.
            steps::build_offset_ball(ball, data, msa_ball_path, msa_data_path)?
        };
        ftlog::info!("Offset Ball has {} leaves.", off_ball.leaves().len());

        // Build the MSA.
        steps::build_aligned(hamming_metric, &args.cost_matrix, &off_ball, &data, msa_fasta_path)?
    };

    ftlog::info!(
        "Finished reading/aligning {} sequences with width = {}.",
        data.cardinality(),
        data.dimensionality_hint().0
    );

    // let ps_quality = data.par_scoring_pairwise(b'-', 1, 1);
    let ps_quality = data.par_scoring_pairwise_subsample(b'-', 1, 1);
    ftlog::info!("Pairwise scoring metric: {ps_quality}");

    Ok(())
}
