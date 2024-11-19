//! Distance functions used in the benchmarks.

/// The distance functions that can be used for nearest neighbors search.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum DistanceFunctions {
    /// Euclidean distance.
    #[clap(name = "euclidean")]
    Euclidean,
    /// Manhattan distance.
    #[clap(name = "manhattan")]
    Manhattan,
    /// Cosine distance.
    #[clap(name = "cosine")]
    Cosine,
    /// Hamming distance.
    #[clap(name = "hamming")]
    Hamming,
    /// Jaccard distance.
    #[clap(name = "jaccard")]
    /// Levenshtein edit distance.
    #[clap(name = "levenshtein")]
    Levenshtein,
}
