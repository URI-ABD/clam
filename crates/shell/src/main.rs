//! CLI for CLAM-MBED, the dimension reduction tool.

mod commands;
mod data;
mod metrics;
mod trees;

use clap::Parser;

use commands::Commands;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The random seed to use.
    #[arg(short('s'), long, default_value = "42")]
    seed: Option<u64>,

    /// The subcommand to run.
    #[command(subcommand)]
    command: Commands,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    println!("Args: {args:?}");

    let seed = args.seed;

    match args.command {
        Commands::Cakes { action } => match action {
            commands::cakes::CakesAction::Build {
                inp_path,
                out_path,
                tree_path,
                metric,
                balanced,
                permuted,
            } => commands::cakes::build_new_tree(
                inp_path,
                out_path,
                tree_path,
                metric.shell_metric(),
                seed,
                balanced,
                permuted,
            )?,
            commands::cakes::CakesAction::Search { .. } => {
                todo!("Tom")
            }
        },
        Commands::Musals { .. } => todo!("Emily"),
    }

    Ok(())
}
