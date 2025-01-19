//! The `Levenshtein` distance metric.

use std::sync::{Arc, RwLock};

use abd_clam::{metric::ParMetric, Metric};
use distances::Number;

use super::{CountingMetric, ParCountingMetric};

/// The `Levenshtein` distance metric.
pub struct Levenshtein(Arc<RwLock<usize>>, bool);

impl Levenshtein {
    /// Creates a new `Levenshtein` distance metric.
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(0)), true)
    }
}

impl Metric<String, u32> for Levenshtein {
    fn distance(&self, a: &String, b: &String) -> u32 {
        if self.1 {
            self.increment();
        }
        stringzilla::sz::edit_distance(a, b).as_u32()
    }

    fn name(&self) -> &str {
        "levenshtein"
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

impl CountingMetric<String, u32> for Levenshtein {
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

impl ParMetric<String, u32> for Levenshtein {}

impl ParCountingMetric<String, u32> for Levenshtein {}
