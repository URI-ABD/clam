//! The commands under the `clam` CLI.

pub mod cakes;
pub mod generate_data;
pub mod mbed;
pub mod musals;

use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// CLAM-Augmented K-nearest-neighbors Entropy-scaling Search
    Cakes {
        #[clap(subcommand)]
        action: cakes::CakesAction,
    },
    /// MUltiple Sequence ALignment at Scale
    Musals {
        #[clap(subcommand)]
        action: musals::MusalsAction,
    },
    /// Dimension Reduction with CLAM-MBED
    Mbed {
        #[clap(subcommand)]
        action: mbed::MbedAction,
    },
    /// Generate synthetic datasets for testing and benchmarking
    GenerateData {
        #[clap(subcommand)]
        action: generate_data::GenerateDataAction,
    },
}
