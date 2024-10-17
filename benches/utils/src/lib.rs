#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::style,
    clippy::complexity,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
//! Utilities for running benchmarks in CLAM.

use ftlog::{
    appender::{FileAppender, Period},
    LevelFilter, LoggerGuard,
};

pub mod ann_benchmarks;
pub mod fasta;
pub mod metrics;
pub mod radio_ml;
pub mod reports;
pub mod types;

pub use metrics::Complex;

/// Configures the logger.
///
/// # Errors
///
/// - If a logs directory could not be located/created.
/// - If the logger could not be initialized.
pub fn configure_logger(file_name: &str) -> Result<(LoggerGuard, std::path::PathBuf), String> {
    let root_dir = std::path::PathBuf::from(".")
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let logs_dir = root_dir.join("logs");
    if !logs_dir.exists() {
        std::fs::create_dir(&logs_dir).map_err(|e| e.to_string())?;
    }
    let log_path = logs_dir.join(format!("{file_name}.log"));

    let writer = FileAppender::builder().path(&log_path).rotate(Period::Day).build();

    let err_path = log_path.with_extension("err.log");

    let guard = ftlog::Builder::new()
        // global max log level
        .max_log_level(LevelFilter::Info)
        // define root appender, pass None would write to stderr
        .root(writer)
        // write `Debug` and higher logs in ftlog::appender to `err_path` instead of `log_path`
        .filter("ftlog::appender", "ftlog-appender", LevelFilter::Debug)
        .appender("ftlog-appender", FileAppender::new(err_path))
        .try_init()
        .map_err(|e| e.to_string())?;

    Ok((guard, log_path))
}

/// The datasets for benchmarking.
#[derive(clap::ValueEnum, Debug, Clone)]
#[allow(non_camel_case_types, clippy::doc_markdown, clippy::module_name_repetitions)]
#[non_exhaustive]
pub enum RawData {
    /// The DeepImage-1B dataset.
    #[clap(name = "deep-image")]
    DeepImage,
    /// The Fashion-MNIST dataset.
    #[clap(name = "fashion-mnist")]
    FashionMNIST,
    /// The GIST dataset.
    #[clap(name = "gist")]
    GIST,
    /// The GloVe 25 dataset.
    #[clap(name = "glove-25")]
    GloVe_25,
    /// The GloVe 50 dataset.
    #[clap(name = "glove-50")]
    GloVe_50,
    /// The GloVe 100 dataset.
    #[clap(name = "glove-100")]
    GloVe_100,
    /// The GloVe 200 dataset.
    #[clap(name = "glove-200")]
    GloVe_200,
    /// The Kosarak dataset.
    #[clap(name = "kosarak")]
    Kosarak,
    /// The LastFM dataset.
    #[clap(name = "lastfm")]
    LastFM,
    /// The MNIST dataset.
    #[clap(name = "mnist")]
    MNIST,
    /// The MovieLens-10M dataset.
    #[clap(name = "movielens")]
    MovieLens,
    /// The NyTimes dataset.
    #[clap(name = "nytimes")]
    NyTimes,
    /// The SIFT-1M dataset.
    #[clap(name = "sift")]
    SIFT,
    /// A Random dataset with the same dimensions as the SIFT dataset.
    #[clap(name = "random")]
    Random,
    /// The Silva-SSU-Ref dataset.
    #[clap(name = "silva-ssu-ref")]
    SilvaSSURef,
    /// The RadioML dataset.
    #[clap(name = "radio-ml")]
    RadioML,
}

impl RawData {
    /// The name of the dataset.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::DeepImage => "deep-image",
            Self::FashionMNIST => "fashion-mnist",
            Self::GIST => "gist",
            Self::GloVe_25 => "glove-25",
            Self::GloVe_50 => "glove-50",
            Self::GloVe_100 => "glove-100",
            Self::GloVe_200 => "glove-200",
            Self::Kosarak => "kosarak",
            Self::LastFM => "lastfm",
            Self::MNIST => "mnist",
            Self::MovieLens => "movielens",
            Self::NyTimes => "nytimes",
            Self::SIFT => "sift",
            Self::Random => "random",
            Self::SilvaSSURef => "silva-SSU-Ref",
            Self::RadioML => "radio-ml",
        }
    }

    /// Read a vector dataset from the given directory.
    ///
    /// The path of the dataset will be inferred from the dataset name and the
    /// given `inp_dir` as `inp_dir/{name}.hdf5`.
    ///
    /// # Errors
    ///
    /// * If the path does not exist.
    /// * If the dataset is not readable.
    /// * If the dataset is not in the expected format.
    pub fn read_vector<P: AsRef<std::path::Path>, T: hdf5::H5Type + Clone>(
        &self,
        inp_dir: &P,
    ) -> Result<ann_benchmarks::AnnDataset<T>, String> {
        let path = inp_dir.as_ref().join(format!("{}.hdf5", self.name()));
        if path.exists() {
            ann_benchmarks::read(&path, self.is_flattened())
        } else {
            Err(format!("Dataset {} not found: {path:?}", self.name()))
        }
    }

    /// Whether the dataset is flattened.
    #[must_use]
    const fn is_flattened(&self) -> bool {
        matches!(self, Self::Kosarak | Self::MovieLens)
    }

    /// Whether the dataset is tabular.
    #[must_use]
    pub const fn is_tabular(&self) -> bool {
        matches!(
            self,
            Self::DeepImage
                | Self::FashionMNIST
                | Self::GIST
                | Self::GloVe_25
                | Self::GloVe_50
                | Self::GloVe_100
                | Self::GloVe_200
                | Self::MNIST
                | Self::NyTimes
                | Self::SIFT
                | Self::Random
        )
    }

    /// Whether the dataset is of member-sets.
    #[must_use]
    pub const fn is_set(&self) -> bool {
        matches!(self, Self::Kosarak | Self::MovieLens | Self::LastFM)
    }

    /// Whether the dataset is of 'omic sequences.
    #[must_use]
    pub const fn is_sequence(&self) -> bool {
        matches!(self, Self::SilvaSSURef)
    }

    /// The name of the metric to use for the dataset.
    #[must_use]
    pub const fn metric(&self) -> &str {
        match self {
            Self::FashionMNIST | Self::GIST | Self::MNIST | Self::SIFT | Self::Random => "euclidean",
            Self::DeepImage
            | Self::GloVe_25
            | Self::GloVe_50
            | Self::GloVe_100
            | Self::GloVe_200
            | Self::LastFM
            | Self::NyTimes => "cosine",
            Self::Kosarak | Self::MovieLens => "jaccard",
            Self::SilvaSSURef => "hamming",
            Self::RadioML => "dtw",
        }
    }
}
