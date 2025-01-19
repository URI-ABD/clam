//! The `Metric` trait is used for all distance computations in CLAM.

use distances::Number;

mod absolute_difference;
mod cosine;
mod euclidean;
mod hypotenuse;
mod manhattan;

pub use absolute_difference::AbsoluteDifference;
pub use cosine::Cosine;
pub use euclidean::Euclidean;
pub use hypotenuse::Hypotenuse;
pub use manhattan::Manhattan;

#[cfg(feature = "msa")]
mod levenshtein;

#[cfg(feature = "msa")]
pub use levenshtein::Levenshtein;

/// The `Metric` trait is used for all distance computations in CLAM.
///
/// # Type Parameters
///
/// - `I`: The type of the items.
/// - `T`: The type of the distance values.
///
/// # Example
///
/// The following is an example of a `Metric` implementation for the Hamming
/// distance between two sequences of bytes. This implementation short-circuits
/// on the length of the shorter sequence.
///
/// ```rust
/// use abd_clam::metric::{Metric, ParMetric};
///
/// struct Hamming;
///
/// // `I` is any type that can be dereferenced to a slice of bytes.
/// impl<I: AsRef<[u8]>> Metric<I, usize> for Hamming {
///     fn distance(&self, a: &I, b: &I) -> usize {
///         // Count the number of positions where the elements are different.
///         a.as_ref().iter().zip(b.as_ref()).filter(|(x, y)| x != y).count()
///     }
///
///     fn name(&self) -> &str {
///         "hamming"
///     }
///
///     fn has_identity(&self) -> bool {
///         // Two sequences are identical if all of their elements are equal.
///         true
///     }
///
///     fn has_non_negativity(&self) -> bool {
///         // The `usize` type is always non-negative.
///         true
///     }
///
///     fn has_symmetry(&self) -> bool {
///         // The Hamming distance is symmetric.
///         true
///     }
///
///     fn obeys_triangle_inequality(&self) -> bool {
///         // The Hamming distance satisfies the triangle inequality.
///         true
///     }
///
///     fn is_expensive(&self) -> bool {
///         // The Hamming distance is not expensive to compute because it is
///         // linear in the length of the sequences.
///         false
///     }
/// }
///
/// // There is no real benefit to parallelizing the counting of different
/// // elements in the Hamming distance.
/// impl<I: AsRef<[u8]> + Send + Sync> ParMetric<I, usize> for Hamming {}
///
/// // Test the Hamming distance.
/// let a = b"hello";
/// let b = b"world";
/// let metric = Hamming;
///
/// assert_eq!(metric.distance(&a, &b), 4);
/// assert_eq!(metric.par_distance(&a, &b), 4);
/// ```
pub trait Metric<I, T: Number> {
    /// Call the metric on two items.
    fn distance(&self, a: &I, b: &I) -> T;

    /// The name of the metric.
    fn name(&self) -> &str;

    /// Whether the metric provides an identity among the items.
    ///
    /// Identity is defined as `d(a, b) = 0` if and only if `a = b`.
    ///
    /// This is used when computing the diagonal of a pairwise distance matrix.
    fn has_identity(&self) -> bool;

    /// Whether the metric only produces non-negative values.
    ///
    /// Non-negativity is defined as `d(a, b) >= 0` for all items `a` and `b`.
    ///
    /// This is the most important property of metrics for use in CLAM.
    fn has_non_negativity(&self) -> bool;

    /// Whether the metric is symmetric.
    ///
    /// Symmetry is defined as `d(a, b) = d(b, a)` for all items `a` and `b`.
    ///
    /// This is used when computing the lower triangle of a pairwise distance
    /// matrix.
    fn has_symmetry(&self) -> bool;

    /// Whether the metric satisfies the triangle inequality.
    ///
    /// The triangle inequality is defined as `d(a, b) + d(b, c) >= d(a, c)` for
    /// all items `a`, `b`, and `c`.
    ///
    /// If the distance function satisfies the triangle inequality, then the
    /// search results from CAKES will have perfect recall.
    fn obeys_triangle_inequality(&self) -> bool;

    /// Whether the metric is expensive to compute.
    ///
    /// We say that a metric is expensive if it costs more than linear time in
    /// the size of the items to compute the distance between two items.
    ///
    /// When using expensive metrics, we use slightly different parallelism in
    /// CLAM.
    fn is_expensive(&self) -> bool;

    /// Whether an item is equal to another item. Items can only be equal if the
    /// metric provides an identity.
    ///
    /// This is a convenience function that checks if the distance between two
    /// items is zero.
    fn is_equal(&self, a: &I, b: &I) -> bool {
        self.has_identity() && self.distance(a, b) == T::ZERO
    }
}

/// Parallel version of [`Metric`](crate::core::metric::Metric).
#[allow(clippy::module_name_repetitions)]
pub trait ParMetric<I: Send + Sync, T: Number>: Metric<I, T> + Send + Sync {
    /// Parallel version of [`Metric::distance`](crate::core::metric::Metric::distance).
    ///
    /// The default implementation calls the non-parallel version of the
    /// distance function.
    ///
    /// This may be used when the distance function itself can be computed with
    /// some parallelism.
    fn par_distance(&self, a: &I, b: &I) -> T {
        self.distance(a, b)
    }
}

impl<I, T: Number> Metric<I, T> for Box<dyn Metric<I, T>> {
    fn distance(&self, a: &I, b: &I) -> T {
        (**self).distance(a, b)
    }

    fn name(&self) -> &str {
        (**self).name()
    }

    fn has_identity(&self) -> bool {
        (**self).has_identity()
    }

    fn has_non_negativity(&self) -> bool {
        (**self).has_non_negativity()
    }

    fn has_symmetry(&self) -> bool {
        (**self).has_symmetry()
    }

    fn obeys_triangle_inequality(&self) -> bool {
        (**self).obeys_triangle_inequality()
    }

    fn is_expensive(&self) -> bool {
        (**self).is_expensive()
    }
}

impl<I, T: Number> Metric<I, T> for Box<dyn ParMetric<I, T>> {
    fn distance(&self, a: &I, b: &I) -> T {
        (**self).distance(a, b)
    }

    fn name(&self) -> &str {
        (**self).name()
    }

    fn has_identity(&self) -> bool {
        (**self).has_identity()
    }

    fn has_non_negativity(&self) -> bool {
        (**self).has_non_negativity()
    }

    fn has_symmetry(&self) -> bool {
        (**self).has_symmetry()
    }

    fn obeys_triangle_inequality(&self) -> bool {
        (**self).obeys_triangle_inequality()
    }

    fn is_expensive(&self) -> bool {
        (**self).is_expensive()
    }
}

impl<I: Send + Sync, T: Number> ParMetric<I, T> for Box<dyn ParMetric<I, T>> {
    fn par_distance(&self, a: &I, b: &I) -> T {
        (**self).par_distance(a, b)
    }
}
