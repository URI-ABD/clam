//! How much the span of child clusters should be reduced compared to their parent cluster.

use crate::{DistanceValue, utils::SizedHeap};

use super::{BipolarSplit, InitialPole};

/// The Span Reduction Factor (SRF) of a `Cluster` controls how much the span of child clusters should be reduced compared to their parent cluster.
///
/// The `span` of a cluster is the distance between any two of its extremal items, e.g. the distance between poles used for partitioning in a binary tree. This
/// can be thought of as an analog to the diameter of a covering sphere in arbitrary metric (or non-metric) spaces. The SRF is the factor by which the span of
/// child clusters should be reduced compared to their parent. For example, `SpanReductionFactor::Two` means that the span of each child cluster should be at
/// most half the span of its parent.
#[must_use]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub enum SpanReductionFactor {
    /// Use a fixed SRF value. This must be in the range (1, ∞). If the value is outside this range, the SRF defaults to `√2`.
    Fixed(f64),
    /// The SRF is `√2`.
    #[default]
    Sqrt2,
    /// The SRF is `2`.
    Two,
    /// The SRF is `e`.
    E,
    /// The SRF is `π`.
    Pi,
    /// The SRF is the golden ratio `φ = (1 + √5) / 2`.
    Phi,
}

impl std::fmt::Display for SpanReductionFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(srf) => write!(f, "Fixed({srf})"),
            Self::Sqrt2 => write!(f, "Sqrt2"),
            Self::Two => write!(f, "Two"),
            Self::E => write!(f, "E"),
            Self::Pi => write!(f, "Pi"),
            Self::Phi => write!(f, "Phi"),
        }
    }
}

impl From<f64> for SpanReductionFactor {
    fn from(value: f64) -> Self {
        // We allow more tolerance when setting the SRF to common constants.
        if (value - core::f64::consts::SQRT_2).abs() < f64::EPSILON.sqrt() {
            Self::Sqrt2
        } else if (value - 2.0).abs() < f64::EPSILON.sqrt() {
            Self::Two
        } else if (value - core::f64::consts::E).abs() < f64::EPSILON.sqrt() {
            Self::E
        } else if (value - core::f64::consts::PI).abs() < f64::EPSILON.sqrt() {
            Self::Pi
        } else if (value - crate::utils::PHI_F64).abs() < f64::EPSILON.sqrt() {
            Self::Phi
        } else if 1.0 < value && value.is_finite() {
            Self::Fixed(value)
        } else {
            Self::Sqrt2 // Default to Sqrt2 if out of range
        }
    }
}

impl From<f32> for SpanReductionFactor {
    fn from(value: f32) -> Self {
        // We allow more tolerance when setting the SRF to common constants.
        if (value - core::f32::consts::SQRT_2).abs() < f32::EPSILON.sqrt() {
            Self::Sqrt2
        } else if (value - 2.0).abs() < f32::EPSILON.sqrt() {
            Self::Two
        } else if (value - core::f32::consts::E).abs() < f32::EPSILON.sqrt() {
            Self::E
        } else if (value - core::f32::consts::PI).abs() < f32::EPSILON.sqrt() {
            Self::Pi
        } else if (value - crate::utils::PHI_F32).abs() < f32::EPSILON.sqrt() {
            Self::Phi
        } else if 1.0 < value && value.is_finite() {
            Self::Fixed(f64::from(value))
        } else {
            Self::Sqrt2 // Default to Sqrt2 if out of range
        }
    }
}

impl SpanReductionFactor {
    /// Returns the maximum allowed child span for a given span from the parent cluster.
    fn max_child_span_for<T: DistanceValue>(&self, parent_span: T) -> T {
        let factor = match self {
            Self::Fixed(srf) => *srf,
            Self::Sqrt2 => core::f64::consts::SQRT_2,
            Self::Two => 2.0,
            Self::E => core::f64::consts::E,
            Self::Pi => core::f64::consts::PI,
            Self::Phi => crate::utils::PHI_F64,
        };
        let parent_span = parent_span.to_f64().unwrap_or_else(|| unreachable!("DistanceValue must be convertible to f64"));
        T::from_f64(parent_span / factor).unwrap_or_else(|| unreachable!("DistanceValue must be convertible from f64"))
    }

    /// Splits the given items.
    pub(crate) fn split<'a, Id, I, T, M>(
        &self,
        metric: &M,
        l_items: &'a mut [(Id, I)],
        r_items: &'a mut [(Id, I)],
        l_distances: Vec<T>,
        r_distances: Vec<T>,
        span: T,
    ) -> Vec<(usize, &'a mut [(Id, I)])>
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
    {
        // Estimate the span of the left and right splits.
        let l_span = span_estimate(&l_distances);
        let r_span = span_estimate(&r_distances);
        // Determine the maximum span of the widest split.
        let max_span = self.max_child_span_for(span);

        // This max-heap will store splits ordered by their size.
        let mut splits = SizedHeap::new(None);
        let nl = l_items.len();
        splits.push(((l_items, l_distances), (l_span, 1)));
        splits.push(((r_items, r_distances), (r_span, 1 + nl)));

        while splits.peek().is_some_and(|(_, (s, _))| *s > max_span) {
            // Pop the widest split
            let ((items, distances), (_, ci)) = splits.pop().unwrap_or_else(|| unreachable!("child_items is not empty"));
            if items.len() < 2 {
                break;
            }

            // Split it again
            let BipolarSplit {
                l_items,
                r_items,
                l_distances,
                r_distances,
                ..
            } = BipolarSplit::new(items, metric, InitialPole::Distances(distances));

            // Get the spans and center indices of the new splits
            let l_span = span_estimate(&l_distances);
            let r_span = span_estimate(&r_distances);
            let lci = ci;
            let rci = ci + l_items.len();

            // Push the new splits back onto the heap
            splits.push(((l_items, l_distances), (l_span, lci)));
            splits.push(((r_items, r_distances), (r_span, rci)));
        }

        splits.take_items().map(|((c_items, _), (_, ci))| (ci, c_items)).collect::<Vec<_>>()
    }

    /// Splits the given items.
    pub(crate) fn par_split<'a, Id, I, T, M>(
        &self,
        metric: &M,
        l_items: &'a mut [(Id, I)],
        r_items: &'a mut [(Id, I)],
        l_distances: Vec<T>,
        r_distances: Vec<T>,
        span: T,
    ) -> Vec<(usize, &'a mut [(Id, I)])>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
    {
        // Estimate the span of the left and right splits.
        let l_span = span_estimate(&l_distances);
        let r_span = span_estimate(&r_distances);
        // Determine the maximum span of the widest split.
        let max_span = self.max_child_span_for(span);

        // This max-heap will store splits ordered by their size.
        let mut splits = SizedHeap::new(None);
        let nl = l_items.len();
        splits.push(((l_items, l_distances), (l_span, 1)));
        splits.push(((r_items, r_distances), (r_span, 1 + nl)));

        while splits.peek().is_some_and(|(_, (s, _))| *s > max_span) {
            // Pop the widest split
            let ((items, distances), (_, ci)) = splits.pop().unwrap_or_else(|| unreachable!("child_items is not empty"));
            if items.len() < 2 {
                break;
            }

            // Split it again
            let BipolarSplit {
                l_items,
                r_items,
                l_distances,
                r_distances,
                ..
            } = BipolarSplit::par_new(items, metric, InitialPole::Distances(distances));

            // Get the spans and center indices of the new splits
            let l_span = span_estimate(&l_distances);
            let r_span = span_estimate(&r_distances);

            // Push the new splits back onto the heap
            let nl = l_items.len();
            splits.push(((l_items, l_distances), (l_span, ci)));
            splits.push(((r_items, r_distances), (r_span, ci + nl)));
        }

        splits.take_items().map(|((c_items, _), (_, ci))| (ci, c_items)).collect::<Vec<_>>()
    }
}

/// Estimates the Span (maximum distance between any two items) of the given items using a heuristic approach.
fn span_estimate<T>(distances: &[T]) -> T
where
    T: DistanceValue,
{
    if distances.is_empty() {
        T::zero()
    } else if distances.len() == 1 {
        distances[0]
    } else {
        // Behold the fancy heuristic!
        distances.iter().max_by_key(|&d| crate::utils::MaxItem((), *d)).map_or_else(T::zero, |&d| d)
    }
}
