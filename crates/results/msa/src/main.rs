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

use abd_clam::{metric::Levenshtein, msa, Cluster, Dataset};
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

    /// Whether to use a balanced partition.
    #[arg(short('b'), long)]
    balanced: bool,

    /// Optional cost of opening a gap.
    #[arg(short('g'), long)]
    gap_open: Option<usize>,

    /// The number of samples to use for the dataset.
    #[arg(short('n'), long)]
    num_samples: Option<usize>,

    /// The cost matrix to use for the alignment.
    #[arg(short('m'), long)]
    cost_matrix: CostMatrix,

    /// Path to the output directory.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,
}

/// The cost matrix to use for the alignment.
#[derive(clap::ValueEnum, Debug, Clone)]
#[allow(non_camel_case_types, clippy::doc_markdown)]
#[non_exhaustive]
pub enum CostMatrix {
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

impl CostMatrix {
    /// Get the cost matrix.
    #[must_use]
    pub fn cost_matrix<T: Number + Neg<Output = T>>(&self, gap_open: Option<usize>) -> msa::CostMatrix<T> {
        match self {
            Self::Default => msa::CostMatrix::default(),
            Self::DefaultAffine => msa::CostMatrix::default_affine(gap_open),
            Self::ExtendedIupac => msa::CostMatrix::extended_iupac(gap_open),
            Self::Blosum62 => msa::CostMatrix::blosum62(gap_open),
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

    let cost_matrix = args.cost_matrix.cost_matrix::<i32>(args.gap_open);
    let aligner = msa::Aligner::new(&cost_matrix, b'-');

    ftlog::info!("Input file: {:?}", fasta_file.raw_path());
    ftlog::info!("Output directory: {:?}", fasta_file.out_dir());

    let data = fasta_file.read::<i32>(args.num_samples, args.pre_aligned, &aligner)?;
    ftlog::info!(
        "Finished reading original dataset: length range = {:?}",
        data.dimensionality_hint()
    );
    let path_manager = PathManager::new(data.name(), fasta_file.out_dir());

    let metric = Levenshtein;

    let msa_fasta_path = path_manager.msa_fasta_path();
    if !msa_fasta_path.exists() {
        let msa_ball_path = path_manager.msa_ball_path();
        let msa_data_path = path_manager.msa_data_path();

        let (off_ball, perm_data) = if msa_ball_path.exists() && msa_data_path.exists() {
            // Read the Offset Ball and the dataset.
            steps::read_permuted_ball(&msa_ball_path, &msa_data_path, &aligner)?
        } else {
            let ball_path = path_manager.ball_path();
            let ball = if ball_path.exists() {
                // Read the Ball.
                steps::read_ball(&ball_path)?
            } else {
                // Build the Ball.
                if args.balanced {
                    steps::build_balanced_ball(&data, &metric, &ball_path, &path_manager.ball_csv_path())?
                } else {
                    steps::build_ball(&data, &metric, &ball_path, &path_manager.ball_csv_path())?
                }
            };
            ftlog::info!("Ball has {} leaves.", ball.leaves().len());

            // Build the Offset Ball and the dataset.
            steps::build_perm_ball(ball, data, &metric, &msa_ball_path, &msa_data_path)?
        };
        ftlog::info!("Offset Ball has {} leaves.", off_ball.leaves().len());

        // Build the MSA.
        steps::build_aligned(&args.cost_matrix, args.gap_open, &off_ball, &perm_data, &msa_fasta_path)?;
        ftlog::info!("Finished building MSA.");
    };

    // Read the aligned sequences and load the data.
    ftlog::info!("Reading aligned sequences from: {msa_fasta_path:?}");
    let msa_data = steps::read_aligned(&msa_fasta_path, &aligner)?;

    ftlog::info!(
        "Finished reading {} aligned sequences with width = {:?}.",
        msa_data.cardinality(),
        msa_data.dimensionality_hint()
    );

    // Compute the quality metrics.

    let gap_char = b'-';
    let gap_penalty = 1;
    let mismatch_penalty = 1;
    let gap_open_penalty = 10;
    let gap_ext_penalty = 1;

    let ps_quality = msa_data.par_scoring_pairwise_subsample(gap_char, gap_penalty, mismatch_penalty);
    ftlog::info!("Pairwise scoring metric estimate: {ps_quality}");

    // let ps_quality = msa_data.par_scoring_pairwise(gap_char, gap_penalty, mismatch_penalty);
    // ftlog::info!("Pairwise scoring metric: {ps_quality}");

    let wps_quality =
        msa_data.par_weighted_scoring_pairwise_subsample(gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty);
    ftlog::info!("Weighted pairwise scoring metric estimate: {wps_quality}");

    // let wps_quality = msa_data.par_weighted_scoring_pairwise(gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty);
    // ftlog::info!("Weighted pairwise scoring metric: {wps_quality}");

    let (avg_p, max_p) = msa_data.par_p_distance_stats_subsample(gap_char);
    ftlog::info!("Pairwise distance stats estimate: avg = {avg_p:.4}, max = {max_p:.4}");

    // let (avg_p, max_p) = msa_data.par_p_distance_stats(gap_char);
    // ftlog::info!("Pairwise distance stats: avg = {avg_p}, max = {max_p}");

    let dd_quality = msa_data.par_distance_distortion_subsample(gap_char);
    ftlog::info!("Distance distortion metric estimate: {dd_quality}");

    // let dd_quality = msa_data.par_distance_distortion(gap_char);
    // ftlog::info!("Distance distortion metric: {dd_quality}");

    ftlog::info!("Finished scoring row-wise.");

    ftlog::info!("Converting to column-major format.");
    let col_msa_data = msa_data.par_change_major();
    ftlog::info!("Finished converting to column-major format.");

    let cs_quality = col_msa_data.par_scoring_columns(gap_char, gap_penalty, mismatch_penalty);
    ftlog::info!("Column scoring metric: {cs_quality}");

    ftlog::info!("Finished scoring column-wise.");

    Ok(())
}
