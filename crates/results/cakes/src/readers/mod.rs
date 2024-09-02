//! Readers for the various file formats of datasets used in benchmarks.

mod greengenes;

use crate::{Co, Queries};

/// The available datasets.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum Datasets {
    /// `GreenGenes` 13.5 dataset.
    #[clap(name = "gg_13_5")]
    GreenGenes13x5,
    /// `GreenGenes` 12.10 dataset.
    #[clap(name = "gg_12_10")]
    GreenGenes12x10,
    /// `GreenGenes` 12.10 pre-aligned dataset.
    #[clap(name = "gg_12_10_aligned")]
    GreenGenes12x10Aligned,
}

impl Datasets {
    /// Reads the dataset from the given path.
    pub fn read_fasta(&self, dir_path: &std::path::Path, num_queries: usize) -> Result<(Co, Queries), String> {
        let path = dir_path.join(self.raw_file());
        greengenes::read(&path, num_queries)
    }

    /// Returns the string name of the dataset.
    pub const fn name(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5",
            Self::GreenGenes12x10 => "gg_12_10",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned",
        }
    }

    /// Returns name of the file containing the raw dataset.
    pub const fn raw_file(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5.fasta",
            Self::GreenGenes12x10 => "gg_12_10.fasta",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.fasta",
        }
    }

    ///  Returns name of the file containing the serialized queries.
    pub const fn queries_file(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5.queries",
            Self::GreenGenes12x10 => "gg_12_10.queries",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.queries",
        }
    }

    /// Returns name of the file to use for the flat-vec dataset.
    pub const fn flat_file(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5.flat_data",
            Self::GreenGenes12x10 => "gg_12_10.flat_data",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.flat_data",
        }
    }

    /// Returns the name of the file to use for the compressed dataset.
    pub const fn compressed_file(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5.codec_data",
            Self::GreenGenes12x10 => "gg_12_10.codec_data",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.codec_data",
        }
    }

    /// Returns the name of the file to use for the ball tree.
    pub const fn ball_file(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5.ball",
            Self::GreenGenes12x10 => "gg_12_10.ball",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.ball",
        }
    }

    /// Returns the name of the file to use for the ball tree.
    pub fn ball_table(&self, postfix: &str) -> String {
        let name = self.name();
        format!("{name}-{postfix}.arrow")
    }

    /// Returns the name of the file to use for the squishy ball tree.
    pub const fn squishy_ball_file(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5.squishy_ball",
            Self::GreenGenes12x10 => "gg_12_10.squishy_ball",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.squishy_ball",
        }
    }
}
