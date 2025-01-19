//! The `DynamicTimeWarping` distance metric.

use std::sync::{Arc, RwLock};

use abd_clam::{metric::ParMetric, Metric};
use bench_utils::{metrics::dtw_distance, Complex};

use super::{CountingMetric, ParCountingMetric};

/// The `DynamicTimeWarping` distance metric.
pub struct DynamicTimeWarping(Arc<RwLock<usize>>, bool);

impl DynamicTimeWarping {
    /// Creates a new `DynamicTimeWarping` distance metric.
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(0)), true)
    }
}

impl<I: AsRef<[Complex<f64>]>> Metric<I, f64> for DynamicTimeWarping {
    fn distance(&self, a: &I, b: &I) -> f64 {
        if self.1 {
            <Self as CountingMetric<I, f64>>::increment(self);
        }
        dtw_distance(a.as_ref(), b.as_ref())
    }

    fn name(&self) -> &str {
        "dtw"
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

impl<I: AsRef<[Complex<f64>]>> CountingMetric<I, f64> for DynamicTimeWarping {
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

impl<I: AsRef<[Complex<f64>]> + Send + Sync> ParMetric<I, f64> for DynamicTimeWarping {
    fn par_distance(&self, a: &I, b: &I) -> f64 {
        if self.1 {
            <Self as CountingMetric<I, f64>>::increment(self);
        }
        dtw_distance(a.as_ref(), b.as_ref())
    }
}

impl<I: AsRef<[Complex<f64>]> + Send + Sync> ParCountingMetric<I, f64> for DynamicTimeWarping {}
