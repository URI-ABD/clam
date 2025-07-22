//! Subcommands for MUSALS

use std::path::PathBuf;

use clap::Subcommand;

/// The MUSALS subcommands for building and evaluating MSAs.
///
/// TODO: Emily
#[derive(Subcommand, Debug)]
pub enum MusalsAction {
    /// Build an MSA and save it to a new file.
    Build {
        /// The path to the input dataset file.
        #[arg(short('i'), long)]
        inp_path: PathBuf,

        /// The path to the output dataset file.
        #[arg(short('o'), long)]
        out_path: PathBuf,
        // TODO Emily: Add or change options as needed
    },
    /// Evaluate the quality of an MSA.
    Evaluate {
        /// The path to the input dataset file.
        #[arg(short('i'), long)]
        inp_path: PathBuf,

        /// The path to the output dataset file.
        #[arg(short('o'), long)]
        out_path: PathBuf,
        // TODO Emily: Add or change options as needed
    },
}
