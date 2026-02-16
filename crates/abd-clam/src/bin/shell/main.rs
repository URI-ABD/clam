//! The main entry point for the CLAM Shell.

use std::path::PathBuf;

use clap::Parser;
use rand::prelude::*;

mod commands;
mod tree;
mod utils;

use commands::Commands;

/// The arguments for the CLAM Shell.
#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path to the input file or directory, depending on the subcommand used.
    #[arg(short('i'), long)]
    inp_path: PathBuf,

    /// The path to the output directory. This will be used differently depending on the subcommand used.
    #[arg(short('o'), long)]
    out_dir: PathBuf,

    /// The random seed to use.
    #[arg(short('s'), long)]
    seed: Option<u64>,

    /// The name of the log-file to use.
    #[arg(short('l'), long, default_value = "shell")]
    log_name: String, // TODO: Write a function to generate a unique log name based on subcommand and timestamp.

    /// The command to run.
    #[command(subcommand)]
    command: Commands,
}

/// The main function for the CLAM Shell.
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Welcome to the CLAM Shell!");

    let args = Args::parse();
    println!("Parsed arguments: {args:?}");

    let (_guard, log_path) = utils::configure_logger(&args.log_name)?;
    ftlog::info!("Log file: {log_path:?}");

    let out_dir = utils::check_path_exists(&args.out_dir, true, true)?;
    ftlog::info!("Output directory: {out_dir:?}");

    // Create the random number generator, using the provided seed if available.
    let mut rng = args.seed.map_or_else(rand::make_rng, rand::rngs::StdRng::seed_from_u64);

    // Execute the command.
    args.command.execute(&args.inp_path, &out_dir, &mut rng)
}
