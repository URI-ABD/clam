//! The Cosine distance function.

use std::sync::{Arc, RwLock};

use abd_clam::{metric::ParMetric, Metric};

use super::{CountingMetric, ParCountingMetric};

/// The Cosine distance function.
pub struct Jaccard(Arc<RwLock<usize>>, bool);

impl Jaccard {
    /// Creates a new `Jaccard` distance metric.
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self(Arc::new(RwLock::new(count)), false)
    }
}

impl Default for Jaccard {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<I: AsRef<[usize]>> Metric<I, f32> for Jaccard {
    fn distance(&self, a: &I, b: &I) -> f32 {
        if self.1 {
            <Self as CountingMetric<I, f32>>::increment(self);
        }
        distances::sets::jaccard(a.as_ref(), b.as_ref())
    }

    fn name(&self) -> &'static str {
        "jaccard"
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

impl<I: AsRef<[usize]>> CountingMetric<I, f32> for Jaccard {
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

impl<I: AsRef<[usize]> + Send + Sync> ParMetric<I, f32> for Jaccard {}

impl<I: AsRef<[usize]> + Send + Sync> ParCountingMetric<I, f32> for Jaccard {}
