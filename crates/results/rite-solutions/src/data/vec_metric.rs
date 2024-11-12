//! Metrics for Vector datasets.

use abd_clam::Metric;
use distances::{number::Float, Number};

use super::wasserstein::wasserstein;

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
    
    /// The Wasserstein distance.
    #[clap(name = "wasserstein")]
    Wasserstein,
}

impl VecMetric {
    /// Returns the metric.
    #[must_use]
    pub fn metric<T: Number, U: Float>(&self) -> Metric<Vec<T>, U> {
        let distance_fn = match self {
            Self::Euclidean => |x: &Vec<T>, y: &Vec<T>| distances::vectors::euclidean::<T, U>(x, y),
            Self::Manhattan => |x: &Vec<T>, y: &Vec<T>| U::from(distances::vectors::manhattan::<T>(x, y)),
            Self::Cosine => |x: &Vec<T>, y: &Vec<T>| distances::vectors::cosine::<T, U>(x, y),
            Self::Wasserstein => |x: &Vec<T>, y: &Vec<T>| U::from(wasserstein::<T, U>(x, y)),
        };
        Metric::new(distance_fn, false)
    }
}
