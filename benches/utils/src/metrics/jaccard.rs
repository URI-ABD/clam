//! The `Jaccard` distance metric.

use abd_clam::{metric::ParMetric, Metric};

/// The `Jaccard` distance metric.
pub struct Jaccard;

impl<I: AsRef<[usize]>> Metric<I, f32> for Jaccard {
    fn distance(&self, a: &I, b: &I) -> f32 {
        distances::sets::jaccard(a.as_ref(), b.as_ref())
    }

    fn name(&self) -> &str {
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

impl<I: AsRef<[usize]> + Send + Sync> ParMetric<I, f32> for Jaccard {}
