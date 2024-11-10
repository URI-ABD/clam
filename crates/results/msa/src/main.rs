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

    /// Whether the original fasta file was pre-aligned by the provider.
    #[arg(short('p'), long)]
    pre_aligned: bool,

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

#[allow(clippy::similar_names)]
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

    let data = fasta_file.read::<i32>(args.num_samples, args.pre_aligned)?;
    ftlog::info!(
        "Finished reading original dataset: length range = {:?}",
        data.dimensionality_hint()
    );
    let path_manager = PathManager::new(data.name(), fasta_file.out_dir());

    let msa_fasta_path = path_manager.msa_fasta_path();
    if !msa_fasta_path.exists() {
        let msa_ball_path = path_manager.msa_ball_path();
        let msa_data_path = path_manager.msa_data_path();

        let (off_ball, perm_data) = if msa_ball_path.exists() && msa_data_path.exists() {
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
        steps::build_aligned(&args.cost_matrix, &off_ball, &perm_data, &msa_fasta_path)?;
        ftlog::info!("Finished building MSA.");
    };

    // Set up the Hamming metric for the aligned sequences and load the data.
    let hamming_fn = |x: &String, y: &String| distances::strings::hamming::<u32>(x, y);
    let hamming_metric = Metric::new(hamming_fn, false);
    ftlog::info!("Reading aligned sequences from: {msa_fasta_path:?}");
    let msa_data = steps::read_aligned(msa_fasta_path, hamming_metric)?;

    ftlog::info!(
        "Finished reading {} aligned sequences with width = {:?}.",
        msa_data.cardinality(),
        msa_data.dimensionality_hint()
    );

    let ps_quality = msa_data.par_scoring_pairwise_subsample(b'-', 1, 1);
    ftlog::info!("Pairwise scoring metric estimate: {ps_quality}");

    // let ps_quality = msa_data.par_scoring_pairwise(b'-', 1, 1);
    // ftlog::info!("Pairwise scoring metric: {ps_quality}");

    let wps_quality = msa_data.par_weighted_scoring_pairwise_subsample(b'-', 10, 1, 10);
    ftlog::info!("Weighted pairwise scoring metric estimate: {wps_quality}");

    // let wps_quality = msa_data.par_weighted_scoring_pairwise(b'-', 10, 1, 10);
    // ftlog::info!("Weighted pairwise scoring metric: {wps_quality}");

    ftlog::info!("Finished scoring row-wise.");

    ftlog::info!("Convert to column-major format.");
    let metric = Metric::default();
    let col_ms_data = msa_data.as_col_major::<Vec<_>>(metric);

    let cs_quality = col_ms_data.par_scoring_columns(b'-', 1, 1);
    ftlog::info!("Column scoring metric estimate: {cs_quality}");

    let wcs_quality = col_ms_data.par_weighted_scoring_columns(b'-', 10, 1, 10);
    ftlog::info!("Weighted column scoring metric estimate: {wcs_quality}");

    ftlog::info!("Finished scoring column-wise.");

    Ok(())
}
