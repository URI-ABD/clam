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

/// Reproducible results for the CAKES and panCAKES papers.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset type
    #[arg(short('d'), long)]
    dataset: data::RawData,

    /// The number of queries to use for benchmarking.
    #[arg(short('q'), long)]
    num_queries: usize,

    /// Path to the input file.
    #[arg(short('i'), long)]
    inp_path: PathBuf,

    /// Path to the output directory.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let log_name = format!("cakes-{}", args.dataset.name());
    let (_guard, log_path) = utils::configure_logger(&log_name)?;
    println!("Log file: {log_path:?}");

    ftlog::info!("{args:?}");

    let inp_path = args.inp_path.canonicalize().map_err(|e| e.to_string())?;

    let out_dir = if let Some(out_dir) = args.out_dir {
        out_dir
    } else {
        ftlog::info!("No output directory specified. Using default.");
        let mut out_dir = inp_path
            .parent()
            .ok_or("No parent directory of `inp_dir`")?
            .to_path_buf();
        out_dir.push(format!("{}_results", args.dataset.name()));
        if !out_dir.exists() {
            std::fs::create_dir(&out_dir).map_err(|e| e.to_string())?;
        }
        out_dir
    }
    .canonicalize()
    .map_err(|e| e.to_string())?;

    ftlog::info!("Input file: {inp_path:?}");
    ftlog::info!("Output directory: {out_dir:?}");

    let mut data = args.dataset.read(&inp_path, &out_dir)?;
    ftlog::info!("Finished reading dataset and queries.");

    // let pool = rayon::ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build()
    //     .map_err(|e| e.to_string())?;

    // pool.install(|| data.benchmark(args.num_queries))?;

    data.benchmark(args.num_queries)?;
    ftlog::info!("Finished benchmarking.");

    Ok(())
}
