//! The Cosine distance function.

use distances::number::Float;

use super::{Metric, ParMetric};

/// The Cosine distance function.
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Deserialize, serde::Serialize)
)]
pub struct Cosine;

impl<I: AsRef<[T]>, T: Float> Metric<I, T> for Cosine {
    fn distance(&self, a: &I, b: &I) -> T {
        distances::vectors::cosine(a.as_ref(), b.as_ref())
    }

    fn name(&self) -> &'static str {
        "cosine"
    }

    fn has_identity(&self) -> bool {
        true
    }

    fn has_non_negativity(&self) -> bool {
        true
    }

    fn has_symmetry(&self) -> bool {
        true
    }

    fn obeys_triangle_inequality(&self) -> bool {
        true
    }

    fn is_expensive(&self) -> bool {
        false
    }
}

impl<I: AsRef<[U]> + Send + Sync, U: Float> ParMetric<I, U> for Cosine {}
