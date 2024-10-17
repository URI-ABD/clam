//! The Dynamic Time Warping distance metric and `Complex` number.

use abd_clam::{metric::ParMetric, Metric};
use distances::number::Float;

/// A complex number
#[derive(Debug, Clone, Copy, bitcode::Decode, bitcode::Encode)]
pub struct Complex<F: Float> {
    /// Real part
    re: F,
    /// Imaginary part
    im: F,
}

impl<F: Float> From<(F, F)> for Complex<F> {
    fn from((re, im): (F, F)) -> Self {
        Self { re, im }
    }
}

impl<F: Float> Complex<F> {
    /// Calculate the magnitude of the complex number
    fn magnitude(&self) -> F {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Calculate the absolute difference between two complex numbers
    fn abs_diff(&self, other: &Self) -> F {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
        .magnitude()
    }
}

/// Calculate Dynamic Time Warping distance between two sequences
#[allow(clippy::module_name_repetitions)]
pub fn dtw_distance<F: Float>(a: &[Complex<F>], b: &[Complex<F>]) -> F {
    // Initialize the DP matrix
    let mut data = vec![vec![F::MAX; b.len() + 1]; a.len() + 1];
    data[0][0] = F::ZERO;

    // Calculate cost matrix
    for ai in 1..=(a.len() + 1) {
        for bi in 1..=(b.len() + 1) {
            let cost = a[ai - 1].abs_diff(&b[bi - 1]);
            data[ai][bi] = cost + F::min(F::min(data[ai - 1][bi], data[ai][bi - 1]), data[ai - 1][bi - 1]);
        }
    }

    // Return final cost
    data[a.len()][b.len()]
}

/// The `Dynamic Time Warping` distance metric.
pub struct Dtw;

impl<I: AsRef<[Complex<f32>]>> Metric<I, f32> for Dtw {
    fn distance(&self, a: &I, b: &I) -> f32 {
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

impl<I: AsRef<[Complex<f32>]> + Send + Sync> ParMetric<I, f32> for Dtw {}

impl<I: AsRef<[Complex<f64>]>> Metric<I, f64> for Dtw {
    fn distance(&self, a: &I, b: &I) -> f64 {
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

impl<I: AsRef<[Complex<f64>]> + Send + Sync> ParMetric<I, f64> for Dtw {}
