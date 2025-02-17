//! CLI for CLAM-MBED, the dimension reduction tool.

use std::path::PathBuf;

use abd_clam::{Ball, FlatVec};
use clap::Parser;

mod distance_functions;
mod quality_measures;
mod workflow;

use distance_functions::DistanceFunction;
use distances::Number;
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

    let out_dir = match args.out_dir {
        Some(out_dir) => {
            let out_dir = out_dir.join(&args.dataset_name);

            if out_dir.exists() {
                let pattern = format!("{}-step-*.npy", args.dataset_name);
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
            .join(&args.dataset_name),
    };

    let file_name = format!("mbed-{}", args.dataset_name);
    let (_guard, log_path) = bench_utils::configure_logger(&file_name, ftlog::LevelFilter::Debug)?;
    ftlog::info!("Logging to: {log_path:?}");

    ftlog::info!("Using {:?} distance function...", args.metric.name());
    let metric = args.metric.metric();

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    match &args.command {
        Commands::Build {
            dimensions,
            checkpoint_frequency,
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
            let name = args.dataset_name.as_str();
            let inp_path = inp_dir.join(format!("{name}.npy"));
            ftlog::info!("Reading data from {inp_path:?}...");
            ftlog::info!("Reducing data to {dimensions} dimensions...");
            ftlog::info!("Saving checkpoints every {checkpoint_frequency} iterations...");
            ftlog::info!("Saving the final result to {name}.npy in {out_dir:?}...");

            let data = FlatVec::<Vec<f64>, usize>::read_npy(&inp_path)?
                .transform_items(|v| v.iter().map(|x| x.as_f32()).collect::<Vec<_>>());
            let criteria = |_: &Ball<f32>| true;
            let reduced_data = workflow::build::<_, _, _, _, _, _, DIM>(
                &out_dir,
                data,
                metric,
                &criteria,
                *dimensions,
                name,
                *checkpoint_frequency,
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

            let data_path = out_dir.join(format!("{name}-reduced.npy"));
            reduced_data.write_npy(&data_path)?;
        }
        Commands::Measure {
            original_data,
            quality_measures,
            exhaustive,
        } => {
            if original_data.extension().is_none_or(|ext| ext != "npy") {
                return Err("Original data must be in .npy format".to_string());
            }

            let inp_path = {
                // The names for the reduced data are "{dataset_name}-step-{step}.npy"
                // where the `step` is the iteration number. The last step is the final
                // result.
                let pattern = format!("{}-step-*.npy", args.dataset_name);
                let mut steps = inp_dir
                    .read_dir()
                    .map_err(|e| format!("Failed to read {inp_dir:?}: {e}"))?
                    .filter_map(|entry| {
                        entry
                            .ok()
                            .and_then(|entry| entry.file_name().into_string().ok())
                            .filter(|name| name == &pattern)
                    })
                    .map(|name| {
                        let i = name.find("step-").unwrap() + "step-".len();
                        let j = name.find(".npy").unwrap();
                        let step_str = &name[i..j];
                        let step = step_str
                            .parse::<u64>()
                            .map_err(|e| format!("Failed to parse step number: {e}"))?;
                        Ok((step, name))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                steps.sort_by_key(|(step, _)| *step);
                let (_, last_step) = steps.pop().ok_or("No steps found")?;
                inp_dir.join(last_step)
            };

            ftlog::info!("Measuring quality of dimension reduction...");
            ftlog::info!("Reading reduced data from {inp_path:?}...");
            ftlog::info!("Reading original data from {original_data:?}...");
            if *exhaustive {
                ftlog::info!("Exhaustively measuring quality using {quality_measures:?}...");
            } else {
                ftlog::info!(
                    "Measuring quality using {:?}...",
                    quality_measures.iter().map(|m| m.name()).collect::<Vec<_>>()
                );
            }
            ftlog::info!("Saving the results in {out_dir:?}...");

            let original_data = FlatVec::<Vec<f64>, usize>::read_npy(&original_data)?
                .transform_items(|v| v.iter().map(|x| x.as_f32()).collect::<Vec<_>>());
            let reduced_data = FlatVec::<[f32; DIM], usize>::read_npy(&inp_path)?;

            let measures = workflow::measure(&original_data, &metric, &reduced_data, quality_measures, *exhaustive);

            for (qm, value) in quality_measures.iter().zip(measures) {
                ftlog::info!("Quality {:?}: {value:.6}", qm.name());
            }
        }
    }

    Ok(())
}
