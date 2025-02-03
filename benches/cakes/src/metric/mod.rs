//! Metrics that count the number of distance computations.

use abd_clam::{metric::ParMetric, Metric};
use distances::Number;

use crate::impl_counting_metric_for_smart_pointer;

mod cosine;
mod dtw;
mod euclidean;
mod hamming;
mod jaccard;
mod levenshtein;
mod macros;

pub use cosine::Cosine;
pub use dtw::DynamicTimeWarping;
pub use euclidean::Euclidean;
pub use hamming::Hamming;
pub use jaccard::Jaccard;
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
    abd_clam::impl_metric_block!();
}

impl<I: Send + Sync, T: Number> Metric<I, T> for Box<dyn ParCountingMetric<I, T>> {
    abd_clam::impl_metric_block!();
}

impl<I: Send + Sync, T: Number> ParMetric<I, T> for Box<dyn ParCountingMetric<I, T>> {
    abd_clam::impl_par_metric_block!();
}

impl<I, T: Number> CountingMetric<I, T> for Box<dyn CountingMetric<I, T>> {
    impl_counting_metric_for_smart_pointer!();
}

impl<I: Send + Sync, T: Number> CountingMetric<I, T> for Box<dyn ParCountingMetric<I, T>> {
    impl_counting_metric_for_smart_pointer!();
}

impl<I: Send + Sync, T: Number> ParCountingMetric<I, T> for Box<dyn ParCountingMetric<I, T>> {}
