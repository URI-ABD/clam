//! The `Euclidean` distance metric.

use std::sync::{Arc, RwLock};

use abd_clam::{metric::ParMetric, Metric};

use super::{CountingMetric, ParCountingMetric};

/// The `Euclidean` distance metric.
pub struct Euclidean(Arc<RwLock<usize>>, bool);

impl Euclidean {
    /// Creates a new `Euclidean` distance metric.
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(0)), true)
    }
}

impl<I: AsRef<[f32]>> Metric<I, f32> for Euclidean {
    fn distance(&self, a: &I, b: &I) -> f32 {
        if self.1 {
            <Self as CountingMetric<I, f32>>::increment(self);
        }
        distances::simd::euclidean_f32(a.as_ref(), b.as_ref())
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

impl<I: AsRef<[f32]>> CountingMetric<I, f32> for Euclidean {
    fn disable_counting(&mut self) {
        self.1 = false;
    }

    fn enable_counting(&mut self) {
        self.1 = true;
    }

    #[allow(clippy::unwrap_used)]
    fn count(&self) -> usize {
        *self.0.read().unwrap()
    }

    #[allow(clippy::unwrap_used)]
    fn reset_count(&self) -> usize {
        let mut count = self.0.write().unwrap();
        let old = *count;
        *count = 0;
        old
    }

    #[allow(clippy::unwrap_used)]
    fn increment(&self) {
        *self.0.write().unwrap() += 1;
    }
}

impl<I: AsRef<[f32]> + Send + Sync> ParMetric<I, f32> for Euclidean {}

impl<I: AsRef<[f32]> + Send + Sync> ParCountingMetric<I, f32> for Euclidean {}

impl<I: AsRef<[f64]>> Metric<I, f64> for Euclidean {
    fn distance(&self, a: &I, b: &I) -> f64 {
        if self.1 {
            <Self as CountingMetric<I, f64>>::increment(self);
        }
        distances::simd::euclidean_f64(a.as_ref(), b.as_ref())
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

impl<I: AsRef<[f64]>> CountingMetric<I, f64> for Euclidean {
    fn disable_counting(&mut self) {
        self.1 = false;
    }

    fn enable_counting(&mut self) {
        self.1 = true;
    }

    #[allow(clippy::unwrap_used)]
    fn count(&self) -> usize {
        *self.0.read().unwrap()
    }

    #[allow(clippy::unwrap_used)]
    fn reset_count(&self) -> usize {
        let mut count = self.0.write().unwrap();
        let old = *count;
        *count = 0;
        old
    }

    #[allow(clippy::unwrap_used)]
    fn increment(&self) {
        *self.0.write().unwrap() += 1;
    }
}

impl<I: AsRef<[f64]> + Send + Sync> ParMetric<I, f64> for Euclidean {}

impl<I: AsRef<[f64]> + Send + Sync> ParCountingMetric<I, f64> for Euclidean {}
