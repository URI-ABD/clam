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

use abd_clam::{metric::Levenshtein, musals, Cluster, Dataset};
use bench_utils::configure_logger;
use clap::Parser;
use distances::Number;

mod path_manager;
mod steps;

/// Reproducible results for the MSA paper.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the input fasta file.
    #[arg(short('i'), long)]
    inp_path: PathBuf,

    /// The number of samples to take from the input dataset. If not provided,
    /// the entire dataset is used.
    #[arg(short('n'), long)]
    num_samples: Option<usize>,

    /// The cost matrix to use for the alignment.
    #[arg(short('m'), long, default_value = "default")]
    matrix: CostMatrix,

    /// Optional cost of opening a gap, if using an affine gap penalty. If not
    /// provided, a flat gap penalty is used.
    #[arg(short('g'), long)]
    gap_open: Option<usize>,

    /// Whether to build a balanced tree.
    #[arg(short('b'), long)]
    balanced: bool,

    /// Path to the output directory.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,

    /// Whether to only compute the quality metrics. This will assume that the
    /// input file contains aligned sequences.
    #[arg(short('q'), long)]
    quality_only: bool,
}

/// The cost matrix to use for the alignment.
#[derive(clap::ValueEnum, Debug, Clone)]
#[allow(non_camel_case_types, clippy::doc_markdown)]
#[non_exhaustive]
pub enum CostMatrix {
    /// The default matrix. All substitutions have a cost of 1.
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
    pub fn cost_matrix<T: Number + core::ops::Neg<Output = T>>(&self, gap_open: Option<usize>) -> musals::CostMatrix<T> {
        match self {
            Self::Default => musals::CostMatrix::default(),
            Self::DefaultAffine => musals::CostMatrix::default_affine(gap_open),
            Self::ExtendedIupac => musals::CostMatrix::extended_iupac(gap_open),
            Self::Blosum62 => musals::CostMatrix::blosum62(gap_open),
        }
    }
}

#[allow(clippy::similar_names, clippy::cognitive_complexity, clippy::too_many_lines)]
fn main() -> Result<(), String> {
    let args = Args::parse();
    println!("{args:?}");

    let fasta_name = args
        .inp_path
        .file_stem()
        .ok_or("No file name found")?
        .to_string_lossy()
        .to_string();
    let log_name = format!("musals-{fasta_name}");
    // We need the `_guard` in scope to ensure proper logging.
    let (_guard, log_path) = configure_logger(&log_name, ftlog::LevelFilter::Info)?;
    println!("Log file: {log_path:?}");
    ftlog::info!("{args:?}");

    let cost_matrix = args.matrix.cost_matrix::<i32>(args.gap_open);
    let aligner = musals::Aligner::new(&cost_matrix, b'-');

    let out_dir = if let Some(out_dir) = args.out_dir {
        if !out_dir.exists() {
            std::fs::create_dir(&out_dir).map_err(|e| e.to_string())?;
        }
        out_dir
    } else {
        args.inp_path.parent().ok_or("No parent directory found")?.to_path_buf()
    };
    ftlog::info!("Input file: {:?}", args.inp_path);
    ftlog::info!("Output directory: {out_dir:?}");

    let (data, _) = bench_utils::fasta::read(&args.inp_path, 0, true)?;
    let msa_data = if args.quality_only {
        steps::read_aligned(&args.inp_path, &aligner)?
    } else {
        let data = if let Some(num_samples) = args.num_samples {
            ftlog::info!("Sub-sampling dataset to {num_samples} samples.");
            let data = data.random_subsample(&mut rand::rng(), num_samples);
            let seq_lens = data.items().iter().map(String::len).collect::<Vec<_>>();
            let (min_len, max_len, _, _, _) = bench_utils::fasta::len_stats(&seq_lens);
            let name = format!("{}-{num_samples}", data.name());
            let data = data
                .with_dim_lower_bound(min_len)
                .with_dim_upper_bound(max_len)
                .with_name(&name);
            ftlog::info!("Sub-sampled dataset: length range = {:?}.", data.dimensionality_hint());
            data
        } else {
            data
        };
        ftlog::info!(
            "Finished reading original dataset: length range = {:?}",
            data.dimensionality_hint()
        );

        let path_manager = path_manager::PathManager::new(data.name(), &out_dir);

        let metric = Levenshtein;

        let msa_fasta_path = path_manager.msa_fasta_path();
        if !msa_fasta_path.exists() {
            // Build the MSA.
            let start = std::time::Instant::now();
            let msa_ball_path = path_manager.msa_ball_path();
            let msa_data_path = path_manager.msa_data_path();

            let (perm_ball, perm_data) = if msa_ball_path.exists() && msa_data_path.exists() {
                // Read the PermutedBall and the dataset.
                steps::read_perm_ball(&msa_ball_path, &msa_data_path)?
            } else {
                let ball_path = path_manager.ball_path();
                let ball = if ball_path.exists() {
                    // Read the Ball.
                    steps::read_ball(&ball_path)?
                } else {
                    steps::build_ball(&data, &metric, &ball_path, &path_manager.ball_csv_path(), args.balanced)?
                };
                ftlog::info!("Ball has {} leaves.", ball.leaves().len());

                // Build the PermutedBall and the dataset.
                steps::build_perm_ball(ball, data, &metric, &msa_ball_path, &msa_data_path)?
            };
            ftlog::info!("Permuted Ball has {} leaves.", perm_ball.leaves().len());

            // Build the MSA.
            steps::build_aligned(&args.matrix, args.gap_open, &perm_ball, &perm_data, &msa_fasta_path)?;
            let elapsed = start.elapsed().as_secs_f32();
            let msa_build_msg = format!("Finished building MSA in {elapsed:.2} seconds.");
            ftlog::info!("{msa_build_msg}");
            println!("{msa_build_msg}");
        }

        // Read the aligned sequences and load the data.
        ftlog::info!("Reading aligned sequences from: {msa_fasta_path:?}");
        steps::read_aligned(&msa_fasta_path, &aligner)?
    };

    ftlog::info!(
        "Finished reading {} aligned sequences with width = {:?}.",
        msa_data.cardinality(),
        msa_data.dimensionality_hint()
    );

    // Compute the quality metrics.
    let start = std::time::Instant::now();
    let gap_penalty = 1;
    let mismatch_penalty = 1;
    let gap_open_penalty = 10;
    let gap_ext_penalty = 1;

    let pg_quality = msa_data.percent_gaps();
    ftlog::info!("Percent gaps: {pg_quality:.4}");

    let ps_quality = msa_data.par_scoring_pairwise_subsample(gap_penalty, mismatch_penalty);
    ftlog::info!("Pairwise scoring metric estimate: {ps_quality}");

    let wps_quality =
        msa_data.par_weighted_scoring_pairwise_subsample(gap_open_penalty, gap_ext_penalty, mismatch_penalty);
    ftlog::info!("Weighted pairwise scoring metric estimate: {wps_quality}");

    let (avg_p, max_p) = msa_data.par_p_distance_stats_subsample();
    ftlog::info!("Pairwise distance stats estimate: avg = {avg_p:.4}, max = {max_p:.4}");

    let dd_quality = msa_data.par_distance_distortion_subsample();
    ftlog::info!("Distance distortion metric estimate: {dd_quality}");

    ftlog::info!("Converting to column-major format.");
    let col_msa_data = msa_data.par_change_major();
    ftlog::info!("Finished converting to column-major format.");

    let cs_quality = col_msa_data.par_scoring_columns(gap_penalty, mismatch_penalty);
    ftlog::info!("Column scoring metric: {cs_quality}");

    let elapsed = start.elapsed().as_secs_f32();
    ftlog::info!("Quality metrics took {elapsed:.2} seconds.");
    println!("Quality metrics took {elapsed:.2} seconds.");

    Ok(())
}
