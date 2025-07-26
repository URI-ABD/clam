//! The commands under the `clam` CLI.

pub mod cakes;
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
}
