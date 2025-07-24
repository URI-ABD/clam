//! The distance functions that may be used on the original data.

use abd_clam::metric::ParMetric;

/// The distance functions that may be used on the original data.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum DistanceFunction {
    /// The Euclidean distance.
    #[clap(name = "euclidean")]
    Euclidean,
    /// The Cosine distance.
    #[clap(name = "cosine")]
    Cosine,
}

impl DistanceFunction {
    /// Get the name of the distance function.
    pub const fn name(&self) -> &str {
        match self {
            Self::Euclidean => "euclidean",
            Self::Cosine => "cosine",
        }
    }

    /// Get the `Metric` for the distance function.
    pub fn metric(&self) -> Box<dyn ParMetric<Vec<f32>, f32>> {
        match self {
            Self::Euclidean => Box::new(abd_clam::metric::Euclidean),
            Self::Cosine => Box::new(abd_clam::metric::Cosine),
        }
    }
}
