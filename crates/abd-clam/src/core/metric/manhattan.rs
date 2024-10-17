//! The `Manhattan` distance metric.

use distances::Number;

use super::{Metric, ParMetric};

/// The `Manhattan` distance metric, also known as the city block distance.
///
/// This is a distance metric that measures the distance between two points in a
/// grid based on the sum of the absolute differences of their coordinates.
pub struct Manhattan;

impl<I: AsRef<[T]>, T: Number> Metric<I, T> for Manhattan {
    fn distance(&self, a: &I, b: &I) -> T {
        distances::vectors::manhattan(a.as_ref(), b.as_ref())
    }

    fn name(&self) -> &str {
        "manhattan"
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

impl<I: AsRef<[T]> + Send + Sync, T: Number> ParMetric<I, T> for Manhattan {}
