//! CLI for CLAM-MBED, the dimension reduction tool.

use std::path::PathBuf;

use abd_clam::{Dataset, FlatVec};
use clap::Parser;

mod dataset;
mod distance_functions;
mod quality_measures;
mod workflow;

use distance_functions::DistanceFunction;
use workflow::Commands;

const DIM: usize = 3;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Args {
    /// Path to the input file.
    #[arg(short('i'), long)]
    inp_dir: PathBuf,

    /// The name of the dataset.
    #[arg(short('n'), long)]
    dataset_name: String,

    /// Path to the output directory. If not provided, the outputs will be
    /// written to the parent directory of the input file.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,

    /// The random seed to use.
    #[arg(short('s'), long, default_value = "42")]
    seed: Option<u64>,

    /// The distance function to use for the original data.
    #[arg(short('m'), long)]
    metric: DistanceFunction,

    /// The subcommand to run.
    #[command(subcommand)]
    command: Commands,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    println!("Args: {args:?}");

    let inp_dir = {
        let inp_dir = args.inp_dir.clone();

        if !inp_dir.exists() {
            return Err(format!("{inp_dir:?} does not exist"));
        }

        if !inp_dir.is_dir() {
            return Err(format!("{inp_dir:?} is not a file"));
        }

        inp_dir
    };

    // let original_data = inp_dir.join(format!("{}.npy", args.dataset_name));
    // if !original_data.exists() {
    //     return Err(format!("{original_data:?} does not exist"));
    // }

    let name = {
        let parts = args.dataset_name.split('.').collect::<Vec<_>>();
        if parts.len() < 2 {
            return Err(format!("Invalid file name: {}", args.dataset_name));
        }
        let n = parts.len() - 1;

        if !["npy", "hdf5"].contains(&parts[n]) {
            return Err(format!("Unsupported file extension: {}", parts[n]));
        }

        parts[..n].join(".")
    };

    let out_dir = match args.out_dir {
        Some(out_dir) => {
            let out_dir = out_dir.join(&name);

            if out_dir.exists() {
                let pattern = format!("{name}-*.npy");
                // delete all files in the output directory that match the pattern
                let files = out_dir
                    .read_dir()
                    .map_err(|e| format!("Failed to read {out_dir:?}: {e}"))?
                    .filter_map(|entry| {
                        entry
                            .ok()
                            .and_then(|entry| entry.file_name().into_string().ok())
                            .filter(|name| name == &pattern)
                    })
                    .collect::<Vec<_>>();
                for file in files {
                    let path = out_dir.join(file);
                    ftlog::info!("Deleting {path:?}...");
                    std::fs::remove_file(&path).map_err(|e| format!("Failed to delete {path:?}: {e}"))?;
                }
            } else {
                ftlog::info!("Creating output directory {out_dir:?}...");
                std::fs::create_dir(&out_dir).map_err(|e| format!("Failed to create {out_dir:?}: {e}"))?;
            }

            if !out_dir.is_dir() {
                return Err(format!("{out_dir:?} is not a directory"));
            }

            out_dir
        }
        None => inp_dir
            .parent()
            .ok_or("Input directory must have a parent directory")?
            .to_path_buf()
            .join(&name),
    };

    let file_name = format!("mbed-{name}");
    let (_guard, log_path) = bench_utils::configure_logger(&file_name, ftlog::LevelFilter::Info)?;
    ftlog::info!("Logging to: {log_path:?}");

    ftlog::info!("Using {:?} distance function...", args.metric.name());
    let metric = args.metric.metric();

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    match &args.command {
        Commands::Build {
            // dimensions,
            // checkpoint_frequency,
            balanced,
            beta,
            k,
            dk,
            f,
            retention_depth,
            dt,
            patience,
            target,
            max_steps,
        } => {
            ftlog::info!("Reducing data to {DIM} dimensions...");
            ftlog::info!("Saving the final result in {out_dir:?}...");

            let data = dataset::read(&inp_dir, &args.dataset_name)?;

            let reduced_data = workflow::build::<_, _, _, _, f32, DIM>(
                &out_dir,
                &data,
                metric,
                *balanced,
                args.seed,
                *beta,
                *k,
                *dk,
                *retention_depth,
                *f,
                *dt,
                *patience,
                *target,
                *max_steps,
            )?;

            let reduced_path = out_dir.join(format!("{}-reduced.npy", data.name()));
            reduced_data.write_npy(&reduced_path)?;
        }
        Commands::Measure {
            quality_measures,
            exhaustive,
        } => {
            ftlog::info!("Measuring quality of dimension reduction...");
            if *exhaustive {
                ftlog::info!("Exhaustively measuring quality using {quality_measures:?}...");
            } else {
                ftlog::info!("Measuring quality using {quality_measures:?}...");
            }

            ftlog::info!("Reading original data from {inp_dir:?}...");
            let original_data = dataset::read(&inp_dir, &args.dataset_name)?;

            let reduced_path = out_dir.join(format!("{}-reduced.npy", original_data.name()));
            if !reduced_path.exists() {
                return Err(format!("{reduced_path:?} does not exist"));
            }

            ftlog::info!("Reading reduced data from {reduced_path:?}...");
            let reduced_data = FlatVec::<[f32; DIM], usize>::read_npy(&reduced_path)?;

            let measures = workflow::measure(&original_data, &metric, &reduced_data, quality_measures, *exhaustive);

            for (qm, value) in quality_measures.iter().zip(measures) {
                let msg = format!("Quality {:?}: {value:.6}", qm.name());
                ftlog::info!("{msg}");
                println!("{msg}");
            }
        }
    }

    Ok(())
}
