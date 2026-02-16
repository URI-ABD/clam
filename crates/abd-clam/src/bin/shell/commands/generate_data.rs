//! Generate random data for tests and benchmarks.

use clap::Subcommand;

/// Generate random data or augment existing data with similar points.
#[derive(Subcommand, Debug)]
pub enum Action {}
