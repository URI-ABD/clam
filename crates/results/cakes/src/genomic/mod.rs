//! Readers for the various file formats of datasets used in benchmarks.

#![allow(dead_code)]

mod greengenes;

use crate::{CoGen, QueriesGen};

/// The available datasets.
#[derive(clap::ValueEnum, Debug, Clone)]
#[allow(clippy::module_name_repetitions)]
pub enum GenomicDataset {
    /// `GreenGenes` 13.5 dataset.
    #[clap(name = "gg_13_5")]
    GreenGenes13x5,
    /// `GreenGenes` 12.10 dataset.
    #[clap(name = "gg_12_10")]
    GreenGenes12x10,
    /// `GreenGenes` 12.10 pre-aligned dataset.
    #[clap(name = "gg_12_10_aligned")]
    GreenGenes12x10Aligned,
    /// The `Silva` dataset.
    #[clap(name = "silva")]
    Silva,
    /// The `PdbSeq` dataset.
    #[clap(name = "pdb-seq")]
    PdbSeq,
}

impl GenomicDataset {
    /// Reads the dataset from the given path.
    pub fn read_fasta(&self, dir_path: &std::path::Path, num_queries: usize) -> Result<(CoGen, QueriesGen), String> {
        let path = dir_path.join(self.raw_file());
        greengenes::read(&path, num_queries)
    }

    /// Returns the string name of the dataset.
    pub const fn name(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5",
            Self::GreenGenes12x10 => "gg_12_10",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned",
            Self::Silva => "silva-SSU-Ref",
            Self::PdbSeq => "pdb_seq_1000max_protein",
        }
    }

    /// Returns name of the file containing the raw dataset.
    pub const fn raw_file(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5.fasta",
            Self::GreenGenes12x10 => "gg_12_10.fasta",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.fasta",
            Self::Silva => "silva-SSU-Ref.fasta",
            Self::PdbSeq => "pdb_seq_1000max_protein.fasta",
        }
    }

    ///  Returns name of the file containing the serialized queries.
    pub const fn queries_file(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5.queries",
            Self::GreenGenes12x10 => "gg_12_10.queries",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.queries",
            Self::Silva => "silva-SSU-Ref.queries",
            Self::PdbSeq => "pdb_seq_1000max_protein.queries",
        }
    }

    /// Returns name of the file to use for the flat-vec dataset.
    pub const fn flat_file(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5.flat_data",
            Self::GreenGenes12x10 => "gg_12_10.flat_data",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.flat_data",
            Self::Silva => "silva-SSU-Ref.flat_data",
            Self::PdbSeq => "pdb_seq_1000max_protein.flat_data",
        }
    }

    /// Returns the name of the file to use for the compressed dataset.
    pub const fn compressed_file(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5.codec_data",
            Self::GreenGenes12x10 => "gg_12_10.codec_data",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.codec_data",
            Self::Silva => "silva-SSU-Ref.codec_data",
            Self::PdbSeq => "pdb_seq_1000max_protein.codec_data",
        }
    }

    /// Returns the name of the file to use for the ball tree.
    pub const fn ball_file(&self) -> &str {
        match self {
            Self::GreenGenes13x5 => "gg_13_5.ball",
            Self::GreenGenes12x10 => "gg_12_10.ball",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.ball",
            Self::Silva => "silva-SSU-Ref.ball",
            Self::PdbSeq => "pdb_seq_1000max_protein.ball",
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
            Self::GreenGenes13x5 => "gg_13_5.squishy_ball",
            Self::GreenGenes12x10 => "gg_12_10.squishy_ball",
            Self::GreenGenes12x10Aligned => "gg_12_10_aligned.squishy_ball",
            Self::Silva => "silva-SSU-Ref.squishy_ball",
            Self::PdbSeq => "pdb_seq_1000max_protein.squishy_ball",
        }
    }
}
