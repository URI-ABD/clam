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

use clap::Parser;

mod data;
mod utils;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset type
    #[arg(short('d'), long)]
    dataset: data::Dataset,

    /// The number of queries to hold out.
    #[arg(short('q'), long)]
    num_queries: usize,

    /// Path to the input directory.
    #[arg(short('i'), long)]
    inp_dir: PathBuf,

    /// Path to the output directory.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    println!("{args:?}");

    let log_name = format!("cakes-{}", args.dataset.name());
    let (_guard, log_path) = utils::configure_logger(&log_name)?;
    println!("Log file: {}", log_path.display());

    ftlog::info!("Hello, world!");
    ftlog::info!("{args:?}");

    Ok(())
}
