//! The workflow for dimensionality reduction using `clam-mbed`.

use clap::Subcommand;

use crate::quality_measures::QualityMeasures;

mod build;
mod measure;

pub use build::build;
pub use measure::measure;

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Create a dimension reduction for the given dataset.
    Build {
        /// Whether to use the balanced ball clustering algorithm.
        #[arg(short('b'), long)]
        balanced: bool,

        /// The damping factor for the mass-spring system.
        #[arg(short('B'), long, default_value = "0.99")]
        beta: f64,

        /// The spring constant for the mass-spring system.
        #[arg(short('k'), long, default_value = "1.0")]
        k: f64,

        /// The factor by which to decrease the spring constant at each
        /// iteration.
        #[arg(short('K'), long, default_value = "0.5")]
        dk: f64,

        /// The time step for each iteration of the mass-spring system.
        #[arg(short('t'), long, default_value = "0.01")]
        dt: f64,

        /// The number of iterations to wait before stopping the optimization if
        /// the stability does not increase.
        #[arg(short('p'), long, default_value = "100")]
        patience: usize,

        /// The target stability value. If the stability reaches this value, the
        /// optimization is stopped.
        #[arg(short('T'), long, default_value = "0.001")]
        target: f64,

        /// The maximum number of iterations to run.
        #[arg(short('M'), long, default_value = "10000")]
        max_steps: usize,
    },
    /// Measure the quality of a dimension reduction.
    Measure {
        /// The quality measures to calculate.
        // #[arg(short('q'), long, default_value = "pairwise,triangle-inequality,angle")]
        #[arg(short('q'), long, default_value = "fnn")]
        quality_measures: Vec<QualityMeasures>,

        /// Whether to exhaustively measure the quality on all possible
        /// combinations of points. Warning: This may take a long time on even
        /// moderately sized datasets.
        #[arg(short('e'), long, default_value = "false")]
        exhaustive: bool,
    },
}
