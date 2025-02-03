//! Distance metrics for benchmarks.

mod cosine;
mod dtw;
mod euclidean;

pub use cosine::Cosine;
pub use dtw::{dtw_distance, Complex, Dtw};
pub use euclidean::Euclidean;
