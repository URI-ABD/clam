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
        /// The path to the output directory.
        #[arg(short('o'), long)]
        out_dir: PathBuf,
        // TODO Emily: Add or change options as needed
    },
    /// Evaluate the quality of an MSA.
    Evaluate {
        /// The path to `out_dir` as specified in the `build` command.
        #[arg(short('o'), long)]
        out_dir: PathBuf,
        // TODO Emily: Add or change options as needed
    },
}
