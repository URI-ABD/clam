//! Metrics for use with the CAKES datasets

use abd_clam::metric::{Cosine, Euclidean, Levenshtein};

/// The available metrics for CAKES datasets.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum Metric {
    #[clap(name = "levenshtein")]
    Levenshtein,
    #[clap(name = "euclidean")]
    Euclidean,
    #[clap(name = "cosine")]
    Cosine,
}

impl Metric {
    /// Get the `Metric` for the distance function.
    pub fn shell_metric(&self) -> ShellMetric {
        match self {
            Self::Levenshtein => ShellMetric::Levenshtein(Levenshtein),
            Self::Euclidean => ShellMetric::Euclidean(Euclidean),
            Self::Cosine => ShellMetric::Cosine(Cosine),
        }
    }
}

/// Metrics supported in the CLI.
pub enum ShellMetric {
    Levenshtein(Levenshtein),
    Euclidean(Euclidean),
    Cosine(Cosine),
}

impl std::fmt::Display for ShellMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Levenshtein(_) => write!(f, "Levenshtein"),
            Self::Euclidean(_) => write!(f, "Euclidean"),
            Self::Cosine(_) => write!(f, "Cosine"),
        }
    }
}
