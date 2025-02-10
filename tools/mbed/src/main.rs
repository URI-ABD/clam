//! CLI for CLAM-MBED, the dimension reduction tool.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

mod distance_functions;
mod quality_measures;

use distance_functions::DistanceFunction;
use quality_measures::QualityMeasures;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Args {
    /// Path to the input file.
    #[arg(short('i'), long)]
    input: PathBuf,

    /// The distance function to use for the original data.
    #[arg(short('m'), long)]
    metric: DistanceFunction,

    /// Path to the output directory. If not provided, the output will be
    /// written to the parent directory of the input file.
    #[arg(short('o'), long)]
    output: Option<PathBuf>,

    /// The subcommand to run.
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Create a dimension reduction for the given dataset.
    Build {
        /// The number of dimensions to reduce to.
        #[arg(short('d'), long, default_value = "3")]
        dimensions: usize,

        /// The name (excluding the extension) of the output file. If not
        /// provided, the name will be generated from the input file.
        #[arg(short('n'), long)]
        name: Option<String>,

        /// The frequency of checkpoints, i.e. how often the current state of
        /// the dimension reduction is saved to disk.
        #[arg(short('c'), long, default_value = "100")]
        checkpoint_frequency: usize,
    },
    /// Measure the quality of a dimension reduction.
    Measure {
        /// Path to the original data file.
        #[arg(short('d'), long)]
        original_data: PathBuf,

        /// The quality measures to calculate.
        #[arg(short('q'), long, default_value = "all")]
        quality_measures: Vec<QualityMeasures>,

        /// Whether to exhaustively measure the quality on all possible
        /// combinations of points. Warning: This may take a long time on even
        /// moderately sized datasets.
        #[arg(short('e'), long, default_value = "false")]
        exhaustive: bool,
    },
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    println!("Args: {args:?}");

    let (inp_path, inp_name) = {
        let inp_path = args.input.clone();

        if !inp_path.exists() {
            return Err(format!("{inp_path:?} does not exist"));
        }

        if !inp_path.is_file() {
            return Err(format!("{inp_path:?} is not a file"));
        }

        let inp_name = args
            .input
            .file_stem()
            .ok_or("Input file must have a name")?
            .to_str()
            .ok_or("Input file name must be valid UTF-8")?;

        (inp_path, inp_name)
    };

    let out_dir = match args.output {
        Some(out_dir) => {
            if !out_dir.exists() {
                ftlog::info!("Creating output directory {out_dir:?}...");
                std::fs::create_dir(&out_dir).map_err(|e| format!("Failed to create {out_dir:?}: {e}"))?;
            }

            if !out_dir.is_dir() {
                return Err(format!("{out_dir:?} is not a directory"));
            }

            out_dir
        }
        None => inp_path
            .parent()
            .ok_or("Input file must have a parent directory")?
            .to_path_buf(),
    };

    let (_guard, log_path) = bench_utils::configure_logger(inp_name)?;
    ftlog::info!("Logging to: {log_path:?}");

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    match &args.command {
        Commands::Build {
            dimensions,
            name,
            checkpoint_frequency,
        } => {
            let name = name.as_deref().unwrap_or(inp_name);
            ftlog::info!("Reading data from {inp_path:?}...");
            ftlog::info!("Using {:?} distance function...", args.metric);
            ftlog::info!("Reducing data to {dimensions} dimensions...");
            ftlog::info!("Saving checkpoints every {checkpoint_frequency} iterations...");
            ftlog::info!("Saving the final result to {name}.npy in {out_dir:?}...");
        }
        Commands::Measure {
            original_data,
            quality_measures,
            exhaustive,
        } => {
            let quality_measures = if quality_measures.contains(&QualityMeasures::All) {
                vec![
                    QualityMeasures::PairwiseDistortion,
                    QualityMeasures::TriangleInequalityDistortion,
                    QualityMeasures::AngleDistortion,
                ]
            } else {
                quality_measures.clone()
            };

            ftlog::info!("Measuring quality of dimension reduction...");
            ftlog::info!("Reading reduced data from {inp_path:?}...");
            ftlog::info!("Reading original data from {original_data:?}...");
            if *exhaustive {
                ftlog::info!("Exhaustively measuring quality using {quality_measures:?}...");
            } else {
                ftlog::info!("Measuring quality using {quality_measures:?}...");
            }
            ftlog::info!("Saving the results in {out_dir:?}...");
        }
    }

    Ok(())
}
