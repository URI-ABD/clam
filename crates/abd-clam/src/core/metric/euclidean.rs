//! The `Euclidean` distance metric.

use distances::number::Float;

use super::{Metric, ParMetric};

/// The `Euclidean` distance metric.
pub struct Euclidean;

impl<I: AsRef<[T]>, T: Float> Metric<I, T> for Euclidean {
    fn distance(&self, a: &I, b: &I) -> T {
        distances::vectors::euclidean(a.as_ref(), b.as_ref())
    }

    fn name(&self) -> &str {
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

impl<I: AsRef<[U]> + Send + Sync, U: Float> ParMetric<I, U> for Euclidean {}
