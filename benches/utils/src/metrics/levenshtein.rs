//! The `Levenshtein` edit distance metric.

use abd_clam::{metric::ParMetric, Metric};
use distances::Number;

/// The `Levenshtein` edit distance metric.
pub struct Levenshtein;

impl<I: AsRef<[u8]>, T: Number> Metric<I, T> for Levenshtein {
    fn distance(&self, a: &I, b: &I) -> T {
        T::from(stringzilla::sz::edit_distance(a.as_ref(), b.as_ref()))
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
        true
    }
}

impl<I: AsRef<[u8]> + Send + Sync, T: Number> ParMetric<I, T> for Levenshtein {}
