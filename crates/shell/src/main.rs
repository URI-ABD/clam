//! CLI for CLAM-MBED, the dimension reduction tool.

mod commands;
mod data;
mod metrics;
mod search;
mod trees;

use std::path::PathBuf;

use abd_clam::FlatVec;
use clap::Parser;

use commands::Commands;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path to the input dataset file (not required for generate-data).
    #[arg(short('i'), long)]
    inp_path: Option<PathBuf>,

    /// The name of the distance metric to use for the CAKES tree.
    #[arg(short('m'), long, default_value = "euclidean")]
    metric: crate::metrics::Metric,

    /// The random seed to use.
    #[arg(short('s'), long, default_value = "42")]
    seed: Option<u64>,

    /// The subcommand to run.
    #[command(subcommand)]
    command: Commands,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let seed = args.seed;

    // Handle generate-data separately since it doesn't need input data
    if let Commands::GenerateData { action } = args.command {
        match action {
            commands::generate_data::GenerateDataAction::Generate {
                num_vectors,
                dimensions,
                filename,
                data_type,
                format,
                partitions,
                min_val,
                max_val,
            } => {
                return commands::generate_data::generate_dataset(
                    num_vectors,
                    dimensions,
                    filename,
                    data_type,
                    format,
                    partitions,
                    min_val,
                    max_val,
                    seed,
                );
            }
        }
    }

    // For other commands, we need input data
    let inp_path = args
        .inp_path
        .ok_or_else(|| "Input path (-i/--inp-path) is required for this command".to_string())?;
    let inp_data = data::read(&inp_path)?;
    let metric = args.metric.shell_metric();

    match args.command {
        Commands::Cakes { action } => match action {
            commands::cakes::CakesAction::Build {
                out_dir,
                balanced,
                permuted,
            } => commands::cakes::build_new_tree(inp_data, metric, seed, balanced, permuted, out_dir)?,
            commands::cakes::CakesAction::Search {
                tree_path,
                instances_path,
                query_algorithms,
                output_path,
            } => commands::cakes::search_tree(
                inp_data,
                tree_path,
                data::read(instances_path)?,
                query_algorithms,
                metric,
                output_path,
            )?,
        },
        Commands::Musals { .. } => todo!("Emily"),
        Commands::Mbed { action } => {
            const DIM: usize = 2; // TODO Najib: figure out how to make this dynamic
            match action {
                commands::mbed::MbedAction::Build {
                    out_dir,
                    balanced,
                    beta,
                    k,
                    dk,
                    dt,
                    patience,
                    target,
                    max_steps,
                } => {
                    let reduced_data_path = out_dir.join("reduced_data.npy");
                    if reduced_data_path.exists() {
                        // If the reduced data file already exists, we delete it to avoid confusion.
                        std::fs::remove_file(&reduced_data_path)
                            .map_err(|e| format!("Failed to remove existing reduced data file: {e}"))?;
                    }
                    let reduced_data = commands::mbed::build_new_embedding::<_, _, DIM>(
                        &out_dir, &inp_data, &metric, balanced, seed, beta, k, dk, dt, patience, target, max_steps,
                    )?;
                    reduced_data.write_npy(&reduced_data_path)?;
                }
                commands::mbed::MbedAction::Evaluate {
                    out_dir,
                    measure,
                    exhaustive,
                } => {
                    let reduced_data_path = out_dir.join("reduced_data.npy");
                    if !reduced_data_path.exists() {
                        return Err(format!(
                            "Reduced data file not found at: {}",
                            reduced_data_path.display()
                        ));
                    }
                    let reduced_data = FlatVec::<[f32; DIM], _>::read_npy(&reduced_data_path)?;
                    let quality = measure.measure(&inp_data, &metric, &reduced_data, exhaustive);
                    let quality_file_path = out_dir.join("quality.txt");
                    std::fs::write(&quality_file_path, quality.to_string())
                        .map_err(|e| format!("Failed to write quality measure to file: {e}"))?;
                }
            }
        }
        Commands::GenerateData { .. } => {
            unreachable!("GenerateData is handled earlier in this function");
        }
    }
    Ok(())
}
