//! Subcommands for CAKES

mod build;
mod search;

use std::path::PathBuf;

use clap::Subcommand;

pub use build::build_new_tree;

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
        /// The path to `out_dir` as used in the `build` command.
        #[arg(short('o'), long)]
        out_dir: PathBuf,
    },
}
