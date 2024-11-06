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

use abd_clam::{msa::CostMatrix, Cluster, Dataset, FlatVec, Metric};
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
    ftlog::info!("Finished reading original dataset.");
    let path_manager = PathManager::new(data.name(), fasta_file.out_dir());

    // Set up the Hamming metric for the aligned sequences.
    let msa_fasta_path = path_manager.msa_fasta_path();
    let hamming_fn = |x: &String, y: &String| distances::strings::hamming::<u32>(x, y);
    let metric = Metric::new(hamming_fn, false);

    // Read or build the data of aligned sequences.
    let data = if msa_fasta_path.exists() {
        ftlog::info!("Reading aligned sequences from {msa_fasta_path:?}");

        let ([aligned_sequences, _], [width, _]) = results_cakes::data::fasta::read(&msa_fasta_path, 0)?;
        let (aligned_sequences, metadata): (Vec<_>, Vec<_>) = aligned_sequences.into_iter().unzip();
        FlatVec::new(aligned_sequences, metric)?
            .with_metadata(&metadata)?
            .with_dim_lower_bound(width)
            .with_dim_upper_bound(width)
    } else {
        let msa_ball_path = path_manager.msa_ball_path();
        let msa_data_path = path_manager.msa_data_path();

        // Read or build the Offset Ball.
        let (off_ball, data) = if msa_ball_path.exists() && msa_data_path.exists() {
            ftlog::info!("Reading MSA ball from {msa_ball_path:?}");
            let off_ball = bincode::deserialize_from(std::fs::File::open(&msa_ball_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;

            ftlog::info!("Reading MSA data from {msa_data_path:?}");
            let data = bincode::deserialize_from(std::fs::File::open(&msa_data_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;

            (off_ball, data)
        } else {
            let ball_path = path_manager.ball_path();
            ftlog::info!("Ball path: {ball_path:?}");
            let ball = if ball_path.exists() {
                ftlog::info!("Reading ball from {ball_path:?}");
                // Deserialize the ball from disk.
                bincode::deserialize_from(std::fs::File::open(&ball_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?
            } else {
                steps::build_ball(&data, ball_path, path_manager.ball_csv_path())?
            };
            ftlog::info!("Finished building/reading ball with {} leaves.", ball.leaves().len());

            steps::build_offset_ball(ball, data, msa_ball_path, msa_data_path)?
        };

        ftlog::info!(
            "Finished adapting/reading Offset Ball with {} leaves.",
            off_ball.leaves().len()
        );

        steps::build_aligned(metric, &args.cost_matrix, &off_ball, &data, msa_fasta_path)?
    };

    ftlog::info!(
        "Finished reading/aligning {} sequences with width = {}.",
        data.cardinality(),
        data.dimensionality_hint().0
    );

    // let ps_metric = data.par_scoring_pairwise(b'-', 1, 1);
    let ps_metric = data.par_scoring_pairwise_subsample(b'-', 1, 1);
    ftlog::info!("Pairwise scoring metric: {ps_metric}");

    Ok(())
}
