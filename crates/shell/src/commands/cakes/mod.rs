//! Subcommands for CAKES

mod build;
mod search;

use std::path::PathBuf;

use clap::Subcommand;

pub use build::build_new_tree;

#[derive(Subcommand, Debug)]
pub enum CakesAction {
    Build {
        /// The path to the input dataset file.
        #[arg(short('i'), long)]
        inp_path: PathBuf,

        /// The path to the output dataset file.
        #[arg(short('o'), long)]
        out_path: PathBuf,

        /// The path to the tree file.
        #[arg(short('t'), long)]
        tree_path: PathBuf,

        /// The name of the metric to use.
        #[arg(short('m'), long, default_value = "euclidean")]
        metric: crate::metrics::Metric,

        /// Whether to build a balanced tree.
        #[arg(short('b'), long, default_value_t = false)]
        balanced: bool,

        /// Whether to permute the dataset.
        #[arg(short('p'), long, default_value_t = true)]
        permuted: bool,
    },
    Search {
        /// The path to the dataset file.
        #[arg(short('i'), long)]
        inp_path: PathBuf,

        /// The path to the tree file.
        #[arg(short('t'), long)]
        tree_path: PathBuf,

        /// The name of the metric to use.
        #[arg(short('m'), long, default_value = "euclidean")]
        metric: crate::metrics::Metric,
    },
}
