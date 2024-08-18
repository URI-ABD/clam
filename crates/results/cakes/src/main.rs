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

use abd_clam::Dataset;
use clap::Parser;
use readers::FlatGenomic;

mod readers;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset type
    #[arg(short, long)]
    dataset: readers::Datasets,

    /// Path to the dataset.
    #[arg(short, long)]
    path: PathBuf,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    // Construct the path for the serialized dataset
    let original_path = args.path.canonicalize().map_err(|e| e.to_string())?;
    let flat_vec_path = original_path.with_extension("flatvec");

    let start = std::time::Instant::now();

    // Read the dataset
    let (data, end) = if flat_vec_path.exists() {
        let data: FlatGenomic =
            bincode::deserialize_from(std::fs::File::open(&flat_vec_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let end = start.elapsed();
        println!(
            "Deserialized {:?} dataset from {flat_vec_path:?} with {} sequences.",
            args.dataset,
            data.cardinality()
        );
        (data, end)
    } else {
        let data = args.dataset.read(&original_path)?;
        let end = start.elapsed();
        println!(
            "Read dataset from {original_path:?} with {} sequences.",
            data.cardinality()
        );
        let serde_start = std::time::Instant::now();
        bincode::serialize_into(std::fs::File::create(&flat_vec_path).map_err(|e| e.to_string())?, &data)
            .map_err(|e| e.to_string())?;
        let serde_end = serde_start.elapsed();
        println!(
            "Serialized dataset to {flat_vec_path:?} in {:.6} seconds.",
            serde_end.as_secs_f64()
        );
        (data, end)
    };

    println!("Elapsed time: {:.6} seconds.", end.as_secs_f64());

    println!(
        "Working with {:?} Dataset with {} sequences in {:?} dims.",
        args.dataset,
        data.cardinality(),
        data.dimensionality_hint()
    );

    Ok(())
}
