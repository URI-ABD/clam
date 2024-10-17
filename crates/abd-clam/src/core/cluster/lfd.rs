//! Helpers for dealing with Local Fractal Dimension (LFD) calculations.

use distances::{number::Addition, Number};

/// Helpers for dealing with Local Fractal Dimension (LFD) calculations.
///
/// The LFD of a `Cluster` is a measure of the fractal dimension of the
/// `Cluster` at a local scale. It is calculated as the logarithm of the ratio
/// of the number of points in the cluster to the number of points within a
/// given scale.
pub struct LFD;

impl LFD {
    /// Calculate LFD from radial distances.
    ///
    /// # Arguments
    ///
    /// * `distances` - The radial distances of the points.
    /// * `half_max` - Half of the maximum distance in `distances`.
    ///
    /// # Returns
    ///
    /// The LFD of the points at the given scale.
    pub fn from_radial_distances<T: Number>(distances: &[T], half_max: T) -> f32 {
        if half_max <= T::EPSILON {
            1.0
        } else {
            let count = distances.iter().filter(|&&d| d <= half_max).count();
            if count == 0 {
                1.0
            } else {
                (distances.len().as_f32() / count.as_f32()).log2()
            }
        }
    }

    /// Calculate LFD from distances and a given scale.
    ///
    /// This is calculated as the ratio of the logarithms of the number of
    /// points in the cluster and the number of points within the given scale to
    /// the logarithm of the two scales.
    ///
    /// # Arguments
    ///
    /// * `distances` - The distances of the points.
    /// * `low_scale` - The scale at which to calculate the LFD.
    /// * `max_scale` - The maximum scale of the cluster.
    pub fn from_distances<T: Number>(distances: &[T], low_scale: T, max_scale: T) -> f32 {
        if low_scale.abs_diff(max_scale) <= T::EPSILON {
            1.0
        } else {
            let low_count = distances.iter().filter(|&&d| d <= low_scale).count();
            if low_count == 0 {
                1.0
            } else {
                let max_count = distances.iter().filter(|&&d| d <= max_scale).count();
                (max_count.as_f32().log2() - low_count.as_f32().log2())
                    / (max_scale.as_f32().log2() - low_scale.as_f32().log2())
            }
        }
    }

    /// Given an `lfd` and `cardinality`, calculates the multiplier to use for
    /// multiplying the `radius` to find `k` points.
    ///
    /// # Arguments
    ///
    /// * `lfd` - The Local Fractal Dimension.
    /// * `cardinality` - The number of points.
    /// * `k` - The number of points we hope to find.
    ///
    /// # Returns
    ///
    /// The multiplier to use for increasing the radius.
    #[must_use]
    pub fn multiplier_for_k(lfd: f32, cardinality: usize, k: usize) -> f32 {
        let ratio = k.as_f32() / cardinality.as_f32();
        if lfd.abs_diff(1.0) <= f32::EPSILON {
            ratio
        } else {
            ratio.powf(1. / (lfd + f32::EPSILON))
        }
    }
}
