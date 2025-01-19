//! Manipulating data with CLAM.

use std::path::{Path, PathBuf};

use abd_clam::FlatVec;
use bio::io::fastq::Result;
use instances::{Aligned, MemberSet, Unaligned};

pub mod instances;

mod aligned;
mod ann_set;
mod unaligned;

/// The types of datasets we can work with.
#[non_exhaustive]
pub enum Tree {
    Unaligned(unaligned::Group),
    Aligned(aligned::Group),
    AnnSet(ann_set::Group),
}

impl Tree {
    /// Creates a new tree for unaligned sequences.
    ///
    /// # Errors
    ///
    /// See `unaligned::Group::new`.
    pub fn new_unaligned(
        name: &str,
        out_dir: &Path,
        data: FlatVec<Unaligned, String>,
        queries: Vec<(String, Unaligned)>,
    ) -> Result<Self, String> {
        let path_manager = PathManager {
            name: name.to_string(),
            out_dir: out_dir.to_path_buf(),
        };
        unaligned::Group::new(path_manager, data, queries).map(Tree::Unaligned)
    }

    /// Creates a new tree for aligned sequences.
    ///
    /// # Errors
    ///
    /// See `aligned::Group::new`.
    pub fn new_aligned(
        name: &str,
        out_dir: &Path,
        data: FlatVec<Aligned, String>,
        queries: Vec<(String, Aligned)>,
    ) -> Result<Self, String> {
        let path_manager = PathManager {
            name: name.to_string(),
            out_dir: out_dir.to_path_buf(),
        };
        aligned::Group::new(path_manager, data, queries).map(Tree::Aligned)
    }

    /// Creates a new tree for ANN sets.
    ///
    /// # Errors
    ///
    /// See `ann_set::Group::new`.
    pub fn new_ann_set(
        name: &str,
        out_dir: &Path,
        data: FlatVec<MemberSet<usize>, usize>,
        queries: Vec<(usize, MemberSet<usize>)>,
        ground_truth: Vec<Vec<(usize, f32)>>,
    ) -> Result<Self, String> {
        let path_manager = PathManager {
            name: name.to_string(),
            out_dir: out_dir.to_path_buf(),
        };
        ann_set::Group::new(path_manager, data, queries, ground_truth).map(Tree::AnnSet)
    }

    /// Benchmarks the tree by searching it with a number of queries.
    ///
    /// This needs a mutable reference because the benchmarks on the aligned
    /// sequences temporarily change the metric from Hamming to Levenshtein.
    ///
    /// # Errors
    ///
    /// - If the times cannot be written to disk.
    pub fn benchmark(&mut self, num_queries: usize) -> Result<(), String> {
        match self {
            Self::Unaligned(group) => group.bench_compressive_search(num_queries),
            Self::Aligned(group) => group.bench_compressive_search(num_queries),
            Self::AnnSet(group) => group.bench_compressive_search(num_queries),
        }
    }
}

/// A helper for managing the names of files on disk.
pub struct PathManager {
    name: String,
    out_dir: PathBuf,
}

impl PathManager {
    /// Creates a new `PathManager`.
    #[allow(dead_code)]
    #[must_use]
    pub fn new<P: AsRef<Path>>(name: &str, out_dir: P) -> Self {
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

    /// The path to the binary file containing the `SquishyBall` tree.
    #[must_use]
    pub fn squishy_ball_path(&self) -> PathBuf {
        let name = format!("{}.squishy_ball", self.name());
        self.out_dir().join(name)
    }

    /// The path to the binary file containing the compressed data.
    #[must_use]
    pub fn compressed_path(&self) -> PathBuf {
        let name = format!("{}.compressed", self.name());
        self.out_dir().join(name)
    }

    /// The path to the binary file containing the queries.
    #[must_use]
    pub fn queries_path(&self) -> PathBuf {
        let name = format!("{}.queries", self.name());
        self.out_dir().join(name)
    }

    /// The path to the binary file containing the ground truth.
    ///
    /// This is only relevant for ANN sets.
    #[must_use]
    pub fn ground_truth_path(&self) -> PathBuf {
        let name = format!("{}.ground_truth", self.name());
        self.out_dir().join(name)
    }

    /// The path to the CSV file containing the `Ball` tree.
    #[must_use]
    pub fn ball_csv_path(&self) -> PathBuf {
        let name = format!("{}_ball.csv", self.name());
        self.out_dir().join(name)
    }

    /// The path to the CSV file containing the `SquishyBall` tree before it is trimmed.
    #[must_use]
    pub fn squishy_ball_csv_path(&self) -> PathBuf {
        let name = format!("{}_pre_trim.csv", self.name());
        self.out_dir().join(name)
    }

    /// Path to json file containing the times taken to search the dataset.
    #[must_use]
    pub fn times_path(&self) -> PathBuf {
        let name = format!("{}_times.json", self.name());
        self.out_dir().join(name)
    }

    /// Path to file containing tree used toe making an MSA.
    #[allow(dead_code)]
    #[must_use]
    pub fn msa_ball_path(&self) -> PathBuf {
        let name = format!("{}_ball.msa", self.name());
        self.out_dir().join(name)
    }

    /// Path to file containing containing the MSA of the dataset.
    #[allow(dead_code)]
    #[must_use]
    pub fn msa_data_path(&self) -> PathBuf {
        let name = format!("{}.msa", self.name());
        self.out_dir().join(name)
    }

    /// Path to file containing the MSA of the dataset in FASTA format.
    #[allow(dead_code)]
    #[must_use]
    pub fn msa_fasta_path(&self) -> PathBuf {
        let name = format!("{}_msa.fasta", self.name());
        self.out_dir().join(name)
    }
}
