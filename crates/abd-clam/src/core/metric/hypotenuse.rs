//! The `Hypotenuse` metric.

use distances::{number::Float, Number};

use super::{Metric, ParMetric};

/// The `Hypotenuse` is just the `Euclidean` distance between two points in 2D
/// space.
pub struct Hypotenuse;

impl<T: Number, U: Float> Metric<(T, T), U> for Hypotenuse {
    fn distance(&self, a: &(T, T), b: &(T, T)) -> U {
        let a = [a.0, a.1];
        let b = [b.0, b.1];
        distances::vectors::euclidean(&a, &b)
    }

    fn name(&self) -> &str {
        "hypotenuse"
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

impl<T: Number, U: Float> ParMetric<(T, T), U> for Hypotenuse {}
