//! The datasets we use for benchmarks.

use std::path::Path;

use abd_clam::{cakes::CodecData, FlatVec};

use crate::instances::{Aligned, MemberSet, Unaligned};

mod ann_benchmarks;
mod fasta;

/// The datasets we use for benchmarks.
#[derive(clap::ValueEnum, Debug, Clone)]
#[allow(non_camel_case_types, clippy::doc_markdown)]
pub enum InputDataset {
    /// The GreenGenes 12.10 dataset.
    #[clap(name = "gg_12_10")]
    GreenGenes_12_10,
    /// The pre-aligned GreenGenes 12.10 dataset.
    #[clap(name = "gg_aligned_12_10")]
    GreenGenesAligned_12_10,
    /// The GreenGenes 13.5 dataset.
    #[clap(name = "gg_13_5")]
    GreenGenes_13_5,
    /// The Silva 18S dataset.
    #[clap(name = "silva_18S")]
    Silva_18S,
    /// The pre-aligned Silva 18S dataset.
    #[clap(name = "silva_aligned_18S")]
    SilvaAligned_18S,
    /// The PDB sequence dataset.
    #[clap(name = "pdb_seq")]
    PdbSeq,
    /// The Kosarak dataset.
    #[clap(name = "kosarak")]
    Kosarak,
    /// The MovieLens-10M dataset.
    #[clap(name = "movielens_10m")]
    MovieLens_10M,
}

impl InputDataset {
    pub const fn name(&self) -> &str {
        match self {
            Self::GreenGenes_12_10 => "gg_12_10",
            Self::GreenGenesAligned_12_10 => "gg_aligned_12_10",
            Self::GreenGenes_13_5 => "gg_13_5",
            Self::Silva_18S => "silva_18S",
            Self::SilvaAligned_18S => "silva_aligned_18S",
            Self::PdbSeq => "pdb_seq",
            Self::Kosarak => "kosarak",
            Self::MovieLens_10M => "movielens_10m",
        }
    }

    /// Reads the dataset from the given path.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to the dataset.
    /// * `holdout`: The number of queries to hold out. Only used for the fasta datasets.
    ///
    /// # Returns
    ///
    /// The dataset and queries, if they were read successfully.
    ///
    /// # Errors
    ///
    /// * If the dataset is not readable.
    /// * If the dataset is not in the expected format.
    pub fn read<P: AsRef<Path>>(self, path: &P, holdout: usize) -> Result<(Uncompressed, Queries), String> {
        match self {
            Self::GreenGenes_12_10 | Self::GreenGenes_13_5 | Self::Silva_18S | Self::PdbSeq => {
                let ([data, queries], [min_len, max_len]) = fasta::read(path, holdout)?;
                let (metadata, data): (Vec<_>, Vec<_>) =
                    data.into_iter().map(|(name, seq)| (name, Unaligned::from(seq))).unzip();
                let queries = queries
                    .into_iter()
                    .map(|(name, seq)| (name, Unaligned::from(seq)))
                    .collect();

                let data = FlatVec::new(data, Unaligned::metric())?
                    .with_metadata(metadata)?
                    .with_dim_lower_bound(min_len)
                    .with_dim_upper_bound(max_len);

                Ok((Uncompressed::Unaligned(data), Queries::Unaligned(queries)))
            }
            Self::GreenGenesAligned_12_10 | Self::SilvaAligned_18S => {
                let ([data, queries], [min_len, max_len]) = fasta::read(path, holdout)?;
                let (metadata, data): (Vec<_>, Vec<_>) =
                    data.into_iter().map(|(name, seq)| (name, Aligned::from(seq))).unzip();
                let queries = queries
                    .into_iter()
                    .map(|(name, seq)| (name, Aligned::from(seq)))
                    .collect();

                let data = FlatVec::new(data, Aligned::metric())?
                    .with_metadata(metadata)?
                    .with_dim_lower_bound(min_len)
                    .with_dim_upper_bound(max_len);

                Ok((Uncompressed::Aligned(data), Queries::Aligned(queries)))
            }
            Self::Kosarak | Self::MovieLens_10M => {
                let data = ann_benchmarks::read::<_, usize>(path, true)?;
                let (data, queries) = (data.train, (data.queries, data.neighbors));

                let metric = MemberSet::metric();
                let data = data.iter().map(MemberSet::<_, f32>::from).collect();
                let data = FlatVec::new(data, metric)?;

                Ok((Uncompressed::Set(data), Queries::Set(queries)))
            }
        }
    }
}

/// Uncompressed datasets for use in benchmarks.
#[allow(dead_code)]
pub enum Uncompressed {
    Aligned(FlatVec<Aligned<u32>, u32, String>),
    Unaligned(FlatVec<Unaligned<u32>, u32, String>),
    Set(FlatVec<MemberSet<usize, f32>, f32, usize>),
}

/// Compressed datasets for use in benchmarks.
#[allow(dead_code)]
pub enum Compressed {
    Aligned(CodecData<Aligned<u32>, u32, String>),
    Unaligned(CodecData<Unaligned<u32>, u32, String>),
    Set(CodecData<MemberSet<usize, f32>, f32, usize>),
}

/// Queries for use in benchmarks.
#[allow(dead_code)]
pub enum Queries {
    Aligned(Vec<(String, Aligned<u32>)>),
    Unaligned(Vec<(String, Unaligned<u32>)>),
    Set(ann_benchmarks::GroundTruth<usize>),
}
