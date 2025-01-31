//! The `Hamming` distance metric.

use abd_clam::{metric::ParMetric, Metric};
use distances::number::UInt;

/// The `Hamming` distance metric.
#[derive(Clone, bitcode::Encode, bitcode::Decode)]
pub struct Hamming;

impl<I: AsRef<str>, T: UInt> Metric<I, T> for Hamming {
    fn distance(&self, a: &I, b: &I) -> T {
        distances::strings::hamming(a.as_ref(), b.as_ref())
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

impl<I: AsRef<str> + Send + Sync, T: UInt> ParMetric<I, T> for Hamming {}
