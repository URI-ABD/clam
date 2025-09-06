//! Subcommands for CAKES

mod build;
mod search;

use std::path::PathBuf;

use clap::Subcommand;

pub use build::build_new_tree;
pub use search::{AlgorithmResult, QueryResult, SearchResults, search_tree};

use crate::search::ShellCakes;

#[derive(Subcommand, Debug)]
pub enum CakesAction {
    /// Build a new CAKES tree from the input dataset. `out_path` must be a directory.
    Build,
    /// Search a CAKES tree with the given queries. `out_path` must be a file with '.json' or '.yaml' extension.
    Search {
        /// The path to the queries file.
        #[arg(short('q'), long)]
        queries_path: PathBuf,

        /// The CAKES algorithms to use for searching.
        #[arg(short('c'), long, value_parser = clap::value_parser!(ShellCakes))]
        cakes_algorithms: Vec<ShellCakes>,
    },
}
