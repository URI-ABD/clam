//! Data and algorithm exploration commands.

mod polar_distances;

use clap::Subcommand;

pub use polar_distances::polar_distances;

#[derive(Subcommand, Debug)]
pub enum ExploreAction {
    /// Distances between poles used for partitioning clusters.
    PolarDistances,
}
