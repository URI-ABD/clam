//! The datasets we use for benchmarks.

/// The datasets we use for benchmarks.
#[derive(clap::ValueEnum, Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum Dataset {
    #[clap(name = "gg_12_10")]
    GreenGenes_12_10,
    #[clap(name = "gg_aligned_12_10")]
    GreenGenesAligned_12_10,
    #[clap(name = "gg_13_5")]
    GreenGenes_13_5,
    #[clap(name = "silva_18S")]
    Silva_18S,
    #[clap(name = "silva_aligned_18S")]
    SilvaAligned_18S,
    #[clap(name = "pdb_seq")]
    PdbSeq,
    #[clap(name = "kosarak")]
    Kosarak,
    #[clap(name = "movielens_10m")]
    MovieLens_10M,
}

impl Dataset {
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
}
