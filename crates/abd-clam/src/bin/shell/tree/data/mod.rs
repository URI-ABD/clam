//! Functions for reading and writing datasets, as well as for generating synthetic datasets.

use rand::prelude::*;

pub mod fasta;
pub mod npy;

/// Supported types of datasets in the CLAM Shell.
#[derive(clap::ValueEnum, Debug, Clone)]
#[non_exhaustive]
pub enum ShellDataType {
    /// `String` data for sequence datasets, read from a `fasta` file.
    #[clap(name = "string")]
    String,
    /// `AlignedSequence` data for sequence datasets, read from a `fasta` file.
    #[clap(name = "aligned")]
    Aligned,
    /// `f64` vector data, read from a `npy` file.
    #[clap(name = "f64")]
    F64,
    /// `f32` vector data, read from a `npy` file.
    #[clap(name = "f32")]
    F32,
}

impl core::fmt::Display for ShellDataType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Self::String => "string",
            Self::Aligned => "aligned",
            Self::F64 => "f64",
            Self::F32 => "f32",
        };
        write!(f, "{name}")
    }
}

/// Shuffles the input data and optionally truncates it to the specified number of samples.
pub fn shuffle_and_truncate<T, R: rand::Rng>(mut data: Vec<T>, rng: &mut R, num_samples: Option<usize>) -> Vec<T> {
    data.shuffle(rng);
    if let Some(num_samples) = num_samples {
        data.truncate(num_samples);
    }
    data
}
