//! Manipulating data with CLAM.

use std::path::{Path, PathBuf};

use abd_clam::FlatVec;
use bio::io::fastq::Result;

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
        data: FlatVec<instances::Unaligned<u32>, u32, String>,
        queries: Vec<(String, instances::Unaligned<u32>)>,
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
        data: FlatVec<instances::Aligned<u32>, u32, String>,
        queries: Vec<(String, instances::Aligned<u32>)>,
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
        data: FlatVec<instances::MemberSet<usize, f32>, f32, usize>,
        queries: Vec<(usize, instances::MemberSet<usize, f32>)>,
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
    /// The name of the dataset.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The directory where the dataset is stored.
    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }

    /// The path to the binary file containing the `Ball` tree.
    pub fn ball_path(&self) -> PathBuf {
        let name = format!("{}.ball", self.name());
        self.out_dir().join(name)
    }

    /// The path to the binary file containing the `SquishyBall` tree.
    pub fn squishy_ball_path(&self) -> PathBuf {
        let name = format!("{}.squishy_ball", self.name());
        self.out_dir().join(name)
    }

    /// The path to the binary file containing the compressed data.
    pub fn compressed_path(&self) -> PathBuf {
        let name = format!("{}.compressed", self.name());
        self.out_dir().join(name)
    }

    /// The path to the binary file containing the queries.
    pub fn queries_path(&self) -> PathBuf {
        let name = format!("{}.queries", self.name());
        self.out_dir().join(name)
    }

    /// The path to the binary file containing the ground truth.
    ///
    /// This is only relevant for ANN sets.
    pub fn ground_truth_path(&self) -> PathBuf {
        let name = format!("{}.ground_truth", self.name());
        self.out_dir().join(name)
    }

    /// The path to the CSV file containing the `Ball` tree.
    pub fn ball_csv_path(&self) -> PathBuf {
        let name = format!("{}_ball.csv", self.name());
        self.out_dir().join(name)
    }

    /// The path to the CSV file containing the `SquishyBall` tree before it is trimmed.
    pub fn pre_trim_csv_path(&self) -> PathBuf {
        let name = format!("{}_pre_trim.csv", self.name());
        self.out_dir().join(name)
    }

    /// The path to the CSV file containing the `SquishyBall` tree after it is trimmed.
    pub fn squishy_csv_path(&self) -> PathBuf {
        let name = format!("{}_squishy.csv", self.name());
        self.out_dir().join(name)
    }

    /// Path to json file containing the times taken to search the dataset.
    pub fn times_path(&self) -> PathBuf {
        let name = format!("{}_times.json", self.name());
        self.out_dir().join(name)
    }
}
