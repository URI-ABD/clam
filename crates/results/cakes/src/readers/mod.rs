//! Readers for the various file formats of datasets used in benchmarks.

use abd_clam::FlatVec;

mod greengenes;

/// A flat genomic dataset.
pub type FlatGenomic = FlatVec<String, u32, String>;

/// The available datasets.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum Datasets {
    /// `GreenGenes` dataset.
    #[clap(name = "greengenes")]
    GreenGenes,
}

impl Datasets {
    /// Reads the dataset from the given path.
    pub fn read(&self, path: &std::path::Path) -> Result<FlatGenomic, String> {
        match self {
            Self::GreenGenes => greengenes::read(path),
        }
    }
}
