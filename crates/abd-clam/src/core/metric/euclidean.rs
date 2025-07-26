//! The `Euclidean` distance metric.

use distances::number::Float;

use super::{Metric, ParMetric};

/// The `Euclidean` distance metric.
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Deserialize, serde::Serialize)
)]
pub struct Euclidean;

impl<I: AsRef<[T]>, T: Float> Metric<I, T> for Euclidean {
    fn distance(&self, a: &I, b: &I) -> T {
        distances::vectors::euclidean(a.as_ref(), b.as_ref())
    }

    fn name(&self) -> &'static str {
        "euclidean"
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

impl<I: AsRef<[T]> + Send + Sync, T: Float> ParMetric<I, T> for Euclidean {}
