//! Distance metrics for benchmarks.

mod hamming;
mod jaccard;
mod levenshtein;

pub use hamming::Hamming;
pub use jaccard::Jaccard;
pub use levenshtein::Levenshtein;
