//! Commands for the CHAODA algorithm.

use clap::Subcommand;

/// The specific action to perform with CHAODA.
#[derive(Subcommand, Debug)]
#[non_exhaustive]
pub enum Action {
    /// Train meta-ml models for a suite of CHAODA graph algorithms on the provided tree(s).
    Train,
    /// Benchmark the trained CHAODA models on the provided tree(s) to measure anomaly detection performance.
    Bench,
}
