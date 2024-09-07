//! Readers for the various file formats of datasets used in benchmarks.

mod h5;

use crate::{CoSet, QueriesSet};

/// The available datasets.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum SetDataset {
    /// `Kosarak` dataset.
    #[clap(name = "kosarak")]
    Kosarak,
    /// `Movielens` dataset.
    #[clap(name = "movielens")]
    Movielens,
}

impl SetDataset {
    /// Reads the dataset from the given path.
    pub fn read_raw(&self, dir_path: &std::path::Path) -> Result<(CoSet, QueriesSet), String> {
        let path = dir_path.join(self.raw_file());
        h5::read(&path)
    }

    /// Returns the string name of the dataset.
    pub const fn name(&self) -> &str {
        match self {
            Self::Kosarak => "kosarak",
            Self::Movielens => "movielens",
        }
    }

    /// Returns name of the file containing the raw dataset.
    pub const fn raw_file(&self) -> &str {
        match self {
            Self::Kosarak => "kosarak-jaccard.hdf5",
            Self::Movielens => "movielens10m-jaccard.hdf5",
        }
    }

    ///  Returns name of the file containing the serialized queries.
    pub const fn queries_file(&self) -> &str {
        match self {
            Self::Kosarak => "kosarak.queries",
            Self::Movielens => "movielens.queries",
        }
    }

    /// Returns name of the file to use for the flat-vec dataset.
    pub const fn flat_file(&self) -> &str {
        match self {
            Self::Kosarak => "kosarak.flat_data",
            Self::Movielens => "movielens.flat_data",
        }
    }

    /// Returns the name of the file to use for the compressed dataset.
    pub const fn compressed_file(&self) -> &str {
        match self {
            Self::Kosarak => "kosarak.codec_data",
            Self::Movielens => "movielens.codec_data",
        }
    }

    /// Returns the name of the file to use for the ball tree.
    pub const fn ball_file(&self) -> &str {
        match self {
            Self::Kosarak => "kosarak.ball",
            Self::Movielens => "movielens.ball",
        }
    }

    /// Returns the name of the file to use for the ball tree.
    pub fn ball_table(&self, postfix: &str, extension: &str) -> String {
        let name = self.name();
        format!("{name}-{postfix}.{extension}")
    }

    /// Returns the name of the file to use for the squishy ball tree.
    pub const fn squishy_ball_file(&self) -> &str {
        match self {
            Self::Kosarak => "kosarak.squishy_ball",
            Self::Movielens => "movielens.squishy_ball",
        }
    }
}
