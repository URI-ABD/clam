//! The distance functions that may be used on the original data.

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
