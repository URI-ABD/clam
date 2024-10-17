//! Metrics for Vector datasets.

use abd_clam::metric::{Cosine, Euclidean, Manhattan, ParMetric};
use distances::number::Float;

#[derive(clap::ValueEnum, Debug, Clone)]
pub enum VecMetric {
    /// The Euclidean (L2) distance.
    #[clap(name = "euclidean")]
    Euclidean,

    /// The Manhattan (L1) distance.
    #[clap(name = "manhattan")]
    Manhattan,

    /// The Cosine (angular) distance.
    #[clap(name = "cosine")]
    Cosine,
}

impl VecMetric {
    /// Returns the metric.
    #[must_use]
    pub fn metric<I: AsRef<[T]> + Send + Sync, T: Float>(&self) -> Box<dyn ParMetric<I, T>> {
        match self {
            Self::Euclidean => Box::new(Euclidean),
            Self::Manhattan => Box::new(Manhattan),
            Self::Cosine => Box::new(Cosine),
        }
    }
}
