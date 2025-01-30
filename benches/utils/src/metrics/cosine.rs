//! The Cosine distance function.

use abd_clam::{metric::ParMetric, Metric};

/// The Cosine distance function.
pub struct Cosine;

impl<I: AsRef<[f32]>> Metric<I, f32> for Cosine {
    fn distance(&self, a: &I, b: &I) -> f32 {
        distances::simd::cosine_f32(a.as_ref(), b.as_ref())
    }

    fn name(&self) -> &'static str {
        "cosine"
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

impl<I: AsRef<[f32]> + Send + Sync> ParMetric<I, f32> for Cosine {}

impl<I: AsRef<[f64]>> Metric<I, f64> for Cosine {
    fn distance(&self, a: &I, b: &I) -> f64 {
        distances::simd::cosine_f64(a.as_ref(), b.as_ref())
    }

    fn name(&self) -> &'static str {
        "cosine"
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

impl<I: AsRef<[f64]> + Send + Sync> ParMetric<I, f64> for Cosine {}
