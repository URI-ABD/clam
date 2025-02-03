//! A helper for managing the names of files on disk.

use std::path::{Path, PathBuf};

/// A helper for managing the names of files on disk.
pub struct PathManager {
    name: String,
    out_dir: PathBuf,
}

impl PathManager {
    /// Creates a new `PathManager`.
    #[must_use]
    pub fn new<P: AsRef<Path>>(name: &str, out_dir: &P) -> Self {
        Self {
            name: name.to_string(),
            out_dir: out_dir.as_ref().to_path_buf(),
        }
    }

    /// The name of the dataset.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The directory where the dataset is stored.
    #[must_use]
    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }

    /// The path to the binary file containing the `Ball` tree.
    #[must_use]
    pub fn ball_path(&self) -> PathBuf {
        let name = format!("{}.ball", self.name());
        self.out_dir().join(name)
    }

    /// The path to the CSV file containing the `Ball` tree.
    #[must_use]
    pub fn ball_csv_path(&self) -> PathBuf {
        let name = format!("{}_ball.csv", self.name());
        self.out_dir().join(name)
    }

    /// Path to file containing tree used toe making an MSA.
    #[must_use]
    pub fn msa_ball_path(&self) -> PathBuf {
        let name = format!("{}_ball.msa", self.name());
        self.out_dir().join(name)
    }

    /// Path to file containing containing the MSA of the dataset.
    #[must_use]
    pub fn msa_data_path(&self) -> PathBuf {
        let name = format!("{}.msa", self.name());
        self.out_dir().join(name)
    }

    /// Path to file containing the MSA of the dataset in FASTA format.
    #[must_use]
    pub fn msa_fasta_path(&self) -> PathBuf {
        let name = format!("{}_msa.fasta", self.name());
        self.out_dir().join(name)
    }
}
