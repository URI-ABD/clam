//! CLI for CLAM-MBED, the dimension reduction tool.

mod commands;
mod data;
mod metrics;
mod trees;
mod utils;

use std::path::PathBuf;

use clap::Parser;

use commands::Commands;

use crate::data::npy;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path to the input dataset file.
    #[arg(short('i'), long)]
    inp_path: PathBuf,

    /// The name of the distance metric to use for the CAKES tree.
    #[arg(short('m'), long, default_value = "euclidean")]
    metric: crate::metrics::Metric,

    /// The random seed to use.
    #[arg(short('s'), long, default_value = "42")]
    seed: Option<u64>,

    /// The name of the log-file to use.
    #[arg(short('l'), long, default_value = "shell.log")]
    log_name: String,

    /// The subcommand to run.
    #[command(subcommand)]
    command: Commands,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let inp_path = args.inp_path;
    let metric = args.metric;

    match args.command {
        Commands::Cakes { action } => match action {
            commands::cakes::CakesAction::Build { out_dir, permuted } => {
                let log_name = format!("cakes-build-{}", args.log_name);
                let (_guard, log_path) = utils::configure_logger(&log_name)?;
                println!("Log file: {log_path:?}");

                let inp_data = data::read(inp_path)?;
                commands::cakes::build_new_tree(inp_data, metric, permuted, out_dir)?
            }
            commands::cakes::CakesAction::Search { .. } => {
                let log_name = format!("cakes-search-{}", args.log_name);
                let (_guard, log_path) = utils::configure_logger(&log_name)?;
                println!("Log file: {log_path:?}");

                todo!("Tom")
            }
        },
        Commands::Musals { .. } => todo!("Emily"),
        Commands::Mbed { action } => {
            let inp_data = data::read(inp_path)?;
            const DIM: usize = 2; // TODO Najib: figure out how to make this dynamic
            match action {
                commands::mbed::MbedAction::Build {
                    out_dir,
                    beta,
                    k,
                    dk,
                    dt,
                    patience,
                    target,
                    max_steps,
                } => {
                    let log_name = format!("mbed-build-{}", args.log_name);
                    let (_guard, log_path) = utils::configure_logger(&log_name)?;
                    println!("Log file: {log_path:?}");

                    let reduced_data_path = out_dir.join("reduced_data.npy");
                    if reduced_data_path.exists() {
                        // If the reduced data file already exists, we delete it to avoid confusion.
                        std::fs::remove_file(&reduced_data_path)
                            .map_err(|e| format!("Failed to remove existing reduced data file: {e}"))?;
                    }
                    let reduced_data = commands::mbed::build_new_embedding::<_, _, DIM>(
                        &out_dir, &inp_data, &metric, beta, k, dk, dt, patience, target, max_steps,
                    )?;
                    npy::write_npy(&reduced_data_path, &reduced_data)?;
                }
                commands::mbed::MbedAction::Evaluate {
                    out_dir,
                    measure,
                    exhaustive,
                } => {
                    let log_name = format!("mbed-evaluate-{}", args.log_name);
                    let (_guard, log_path) = utils::configure_logger(&log_name)?;
                    println!("Log file: {log_path:?}");

                    let reduced_data_path = out_dir.join("reduced_data.npy");
                    if !reduced_data_path.exists() {
                        return Err(format!(
                            "Reduced data file not found at: {}",
                            reduced_data_path.display()
                        ));
                    }
                    let reduced_data = npy::read_npy_n::<_, f32, DIM>(&reduced_data_path)?;

                    let quality = measure.measure(&inp_data, &metric, &reduced_data, exhaustive);
                    let quality_file_path = out_dir.join("quality.txt");
                    std::fs::write(&quality_file_path, format!("{}: {quality:.2e}\n", measure.name()))
                        .map_err(|e| format!("Failed to write quality measure to file: {e}"))?;
                }
            }
        }
    }

    Ok(())
}
