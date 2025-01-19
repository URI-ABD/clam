//! Metrics that count the number of distance computations.

use abd_clam::{metric::ParMetric, Metric};
use distances::Number;

mod cosine;
mod dtw;
mod euclidean;
mod hamming;
mod levenshtein;

pub use cosine::Cosine;
pub use dtw::DynamicTimeWarping;
pub use euclidean::Euclidean;
pub use hamming::Hamming;
pub use levenshtein::Levenshtein;

/// A metric that counts the number of distance computations.
#[allow(clippy::module_name_repetitions)]
pub trait CountingMetric<I, T: Number>: Metric<I, T> {
    /// Disables counting the number of distance computations.
    fn disable_counting(&mut self);

    /// Enables counting the number of distance computations.
    fn enable_counting(&mut self);

    /// Returns the number of distance computations made.
    fn count(&self) -> usize;

    /// Resets the counter and returns the previous value.
    fn reset_count(&self) -> usize;

    /// Increments the counter.
    fn increment(&self);
}

/// Parallel version of the `CountingMetric` trait.
#[allow(clippy::module_name_repetitions)]
pub trait ParCountingMetric<I: Send + Sync, T: Number>: ParMetric<I, T> + CountingMetric<I, T> {}

impl<I, T: Number> Metric<I, T> for Box<dyn CountingMetric<I, T>> {
    fn distance(&self, a: &I, b: &I) -> T {
        self.as_ref().distance(a, b)
    }

    fn name(&self) -> &str {
        self.as_ref().name()
    }

    fn has_identity(&self) -> bool {
        self.as_ref().has_identity()
    }

    fn has_non_negativity(&self) -> bool {
        self.as_ref().has_non_negativity()
    }

    fn has_symmetry(&self) -> bool {
        self.as_ref().has_symmetry()
    }

    fn obeys_triangle_inequality(&self) -> bool {
        self.as_ref().obeys_triangle_inequality()
    }

    fn is_expensive(&self) -> bool {
        self.as_ref().is_expensive()
    }
}

impl<I: Send + Sync, T: Number> Metric<I, T> for Box<dyn ParCountingMetric<I, T>> {
    fn distance(&self, a: &I, b: &I) -> T {
        self.as_ref().distance(a, b)
    }

    fn name(&self) -> &str {
        self.as_ref().name()
    }

    fn has_identity(&self) -> bool {
        self.as_ref().has_identity()
    }

    fn has_non_negativity(&self) -> bool {
        self.as_ref().has_non_negativity()
    }

    fn has_symmetry(&self) -> bool {
        self.as_ref().has_symmetry()
    }

    fn obeys_triangle_inequality(&self) -> bool {
        self.as_ref().obeys_triangle_inequality()
    }

    fn is_expensive(&self) -> bool {
        self.as_ref().is_expensive()
    }
}

impl<I: Send + Sync, T: Number> ParMetric<I, T> for Box<dyn ParCountingMetric<I, T>> {}

impl<I, T: Number> CountingMetric<I, T> for Box<dyn CountingMetric<I, T>> {
    fn disable_counting(&mut self) {
        self.as_mut().disable_counting();
    }

    fn enable_counting(&mut self) {
        self.as_mut().enable_counting();
    }

    fn count(&self) -> usize {
        self.as_ref().count()
    }

    fn reset_count(&self) -> usize {
        self.as_ref().reset_count()
    }

    fn increment(&self) {
        self.as_ref().increment();
    }
}

impl<I: Send + Sync, T: Number> CountingMetric<I, T> for Box<dyn ParCountingMetric<I, T>> {
    fn disable_counting(&mut self) {
        self.as_mut().disable_counting();
    }

    fn enable_counting(&mut self) {
        self.as_mut().enable_counting();
    }

    fn count(&self) -> usize {
        self.as_ref().count()
    }

    fn reset_count(&self) -> usize {
        self.as_ref().reset_count()
    }

    fn increment(&self) {
        self.as_ref().increment();
    }
}

impl<I: Send + Sync, T: Number> ParCountingMetric<I, T> for Box<dyn ParCountingMetric<I, T>> {}
