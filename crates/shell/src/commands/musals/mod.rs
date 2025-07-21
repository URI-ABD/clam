//! Subcommands for MUSALS

use std::path::PathBuf;

use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub enum MusalsAction {
    Build {
        /// The path to the input dataset file.
        #[arg(short('i'), long)]
        inp_path: PathBuf,

        /// The path to the output dataset file.
        #[arg(short('o'), long)]
        out_path: PathBuf,
        // TODO Emily: Add more options as needed
    },
    Evaluate {
        /// The path to the input dataset file.
        #[arg(short('i'), long)]
        inp_path: PathBuf,

        /// The path to the output dataset file.
        #[arg(short('o'), long)]
        out_path: PathBuf,
        // TODO Emily: Add more options as needed
    },
}
