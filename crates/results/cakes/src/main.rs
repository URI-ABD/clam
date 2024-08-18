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
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
#![doc = include_str!("../README.md")]

use std::path::PathBuf;

use clap::Parser;

mod readers;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset type
    #[arg(short, long)]
    dataset: readers::Datasets,

    /// Name of the person to greet
    #[arg(short, long)]
    path: PathBuf,

    /// Number of times to greet
    #[arg(short, long, default_value_t = 10)]
    num_samples: usize,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    // Check that the path exists
    if !args.path.exists() {
        return Err(format!("Path {:?} does not exist!", args.path));
    }

    // Read the dataset
    args.dataset.read(&args.path, args.num_samples)?;

    Ok(())
}
