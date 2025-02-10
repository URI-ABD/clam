//! CLI for CLAM-MBED, the dimension reduction tool.

use std::path::PathBuf;

use abd_clam::{Ball, FlatVec, ParDiskIO};
use clap::Parser;

mod distance_functions;
mod quality_measures;
mod workflow;

use distance_functions::DistanceFunction;
use distances::Number;
use workflow::Commands;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Args {
    /// Path to the input file.
    #[arg(short('i'), long)]
    input: PathBuf,

    /// Path to the output directory. If not provided, the outputs will be
    /// written to the parent directory of the input file.
    #[arg(short('o'), long)]
    output: Option<PathBuf>,

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

    let file_name = format!("mbed-{inp_name}");
    let (_guard, log_path) = bench_utils::configure_logger(&file_name, ftlog::LevelFilter::Debug)?;
    ftlog::info!("Logging to: {log_path:?}");

    ftlog::info!("Using {:?} distance function...", args.metric.name());
    let metric = args.metric.metric();

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    match &args.command {
        Commands::Build {
            dimensions,
            name,
            checkpoint_frequency,
            beta,
            k,
            f,
            min_k,
            dt,
            patience,
            target,
            max_steps,
        } => {
            if inp_path.extension().is_none_or(|ext| ext != "npy") {
                return Err("Input file must be in .npy format".to_string());
            }

            let name = name.as_deref().unwrap_or(inp_name);
            ftlog::info!("Reading data from {inp_path:?}...");
            ftlog::info!("Reducing data to {dimensions} dimensions...");
            ftlog::info!("Saving checkpoints every {checkpoint_frequency} iterations...");
            ftlog::info!("Saving the final result to {name}.npy in {out_dir:?}...");

            let data = FlatVec::<Vec<f64>, usize>::read_npy(&inp_path)?
                .transform_items(|v| v.iter().map(|x| x.as_f32()).collect::<Vec<_>>());
            let criteria = |_: &Ball<f32>| true;
            let tree = workflow::build::<_, _, _, _, _, _, 3>(
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
                *f,
                *min_k,
                *dt,
                *patience,
                *target,
                *max_steps,
            )?;

            let tree_path = out_dir.join(format!("{name}-tree.bin"));
            tree.par_write_to(&tree_path)?;
        }
        Commands::Measure {
            original_data,
            quality_measures,
            exhaustive,
        } => {
            if original_data.extension().is_none_or(|ext| ext != "npy") {
                return Err("Original data must be in .npy format".to_string());
            }

            if !inp_name.ends_with("-tree") {
                return Err("Input file must be a tree".to_string());
            }

            if inp_path.extension().is_none_or(|ext| ext != "bin") {
                return Err("Input file must be in .bin format".to_string());
            }

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
            let tree = workflow::Tree::<Ball<f32>, 3>::par_read_from(&inp_path)?;
            let reduced_data = tree.dataset();

            let measures = workflow::measure(&original_data, &metric, reduced_data, quality_measures, *exhaustive);

            for (qm, value) in quality_measures.iter().zip(measures) {
                ftlog::info!("Quality {:?}: {value:.6}", qm.name());
            }
        }
    }

    Ok(())
}
