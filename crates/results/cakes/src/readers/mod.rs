//! Readers for the various file formats of datasets used in benchmarks.

mod greengenes;

/// The available datasets.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum Datasets {
    /// `GreenGenes` dataset.
    #[clap(name = "greengenes")]
    GreenGenes,
}

impl Datasets {
    /// Reads the dataset from the given path.
    pub fn read(&self, path: &std::path::Path, num_samples: usize) -> Result<(), String> {
        match self {
            Self::GreenGenes => greengenes::read(path, num_samples),
        }
    }
}
