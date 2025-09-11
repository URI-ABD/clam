//! Subcommands for CAKES

mod build;
mod search;

use std::path::PathBuf;

use clap::Subcommand;

pub use build::build_new_tree;
pub use search::search_tree;

#[derive(Subcommand, Debug)]
pub enum CakesAction {
    Build {
        /// The path to the output directory. We will construct the output paths
        /// from this directory as we need them.
        #[arg(short('o'), long)]
        out_dir: PathBuf,

        /// Whether to build a balanced tree.
        #[arg(short('b'), long, default_value_t = false)]
        balanced: bool,

        /// Whether to permute the dataset.
        #[arg(short('p'), long, default_value_t = true)]
        permuted: bool,
    },
    Search {
        /// The path to the tree file.
        #[arg(short('t'), long)]
        tree_path: PathBuf,

        #[arg(short('I'), long)]
        instances_path: PathBuf,

        #[arg(short('q'), long, value_parser = clap::value_parser!(crate::search::QueryAlgorithm<f64>))]
        query_algorithms: Vec<crate::search::QueryAlgorithm<f64>>,

        /// The path to the output file for search results (format determined by extension: .json, .yaml, or .yml).
        #[arg(short('o'), long)]
        output_path: PathBuf,
    },
}
