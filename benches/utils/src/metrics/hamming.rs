//! The `Hamming` distance metric.

use abd_clam::{metric::ParMetric, Metric};
use distances::number::Int;

/// The `Hamming` distance metric.
pub struct Hamming<T>(std::marker::PhantomData<T>);

impl<T> Default for Hamming<T> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<I: AsRef<[T]>, T: Int> Metric<I, f32> for Hamming<T> {
    fn distance(&self, a: &I, b: &I) -> f32 {
        distances::sets::jaccard(a.as_ref(), b.as_ref())
    }

    fn name(&self) -> &'static str {
        "hamming"
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

impl<I: AsRef<[T]> + Send + Sync, T: Int> ParMetric<I, f32> for Hamming<T> {}
