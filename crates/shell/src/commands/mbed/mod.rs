//! Subcommands for `Mbed`

mod build;
mod evaluate;

use std::path::PathBuf;

use clap::Subcommand;

pub use build::build_new_embedding;

#[derive(Subcommand, Debug)]
pub enum MbedAction {
    Build {
        /// The path to the output directory.
        #[arg(short('o'), long)]
        out_dir: PathBuf,

        /// The damping factor for the mass-spring system.
        #[arg(short('B'), long, default_value = "0.99")]
        beta: f64,

        /// The spring constant for the mass-spring system.
        #[arg(short('k'), long, default_value = "-.01")]
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
    Evaluate {
        /// The path to `out_dir` used in the `build` command.
        #[arg(short('o'), long)]
        out_dir: PathBuf,

        /// The name of the quality measure to use.
        #[arg(short('M'), long, default_value = "fnn")]
        measure: evaluate::QualityMeasure,

        /// Whether to exhaustively measure the quality on all possible
        /// combinations of points. Warning: This may take a long time on even
        /// moderately sized datasets.
        #[arg(short('e'), long, default_value_t = false)]
        exhaustive: bool,
    },
}
