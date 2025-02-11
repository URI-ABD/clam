//! The workflow for dimensionality reduction using `clam-mbed`.

use std::path::PathBuf;

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
        /// The number of dimensions to reduce to.
        #[arg(short('d'), long, default_value = "3")]
        dimensions: usize,

        /// The name (excluding the extension) of the output file. If not
        /// provided, the name will be generated from the input file.
        #[arg(short('n'), long)]
        name: Option<String>,

        /// The frequency of checkpoints, i.e. how often the current state of
        /// the dimension reduction is saved to disk.
        #[arg(short('c'), long, default_value = "100")]
        checkpoint_frequency: usize,

        /// The damping factor for the mass-spring system.
        #[arg(short('b'), long, default_value = "0.1")]
        beta: f32,

        /// The spring constant for the mass-spring system.
        #[arg(short('k'), long, default_value = "1.0")]
        k: f32,

        /// The factor by which spring constants are multiplied while descending
        /// the tree.
        #[arg(short('f'), long, default_value = "0.9")]
        f: f32,

        /// The minimum spring constant before removing a spring.
        #[arg(short('K'), long, default_value = "0.5")]
        min_k: Option<f32>,

        /// The time step for each iteration of the mass-spring system.
        #[arg(short('t'), long, default_value = "0.001")]
        dt: f32,

        /// The number of iterations to wait before stopping the optimization if
        /// the stability does not increase.
        #[arg(short('p'), long, default_value = "10")]
        patience: usize,

        /// The target stability value. If the stability reaches this value, the
        /// optimization is stopped.
        #[arg(short('T'), long, default_value = "0.99998")]
        target: Option<f32>,

        /// The maximum number of iterations to run.
        #[arg(short('M'), long, default_value = "10000")]
        max_steps: Option<usize>,
    },
    /// Measure the quality of a dimension reduction.
    Measure {
        /// Path to the original data file.
        #[arg(short('d'), long)]
        original_data: PathBuf,

        /// The quality measures to calculate.
        // #[arg(short('q'), long, default_value = "pairwise,triangle-inequality,angle")]
        #[arg(short('q'), long, default_value = "pairwise")]
        quality_measures: Vec<QualityMeasures>,

        /// Whether to exhaustively measure the quality on all possible
        /// combinations of points. Warning: This may take a long time on even
        /// moderately sized datasets.
        #[arg(short('e'), long, default_value = "false")]
        exhaustive: bool,
    },
}
