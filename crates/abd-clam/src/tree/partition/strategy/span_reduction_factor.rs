//! How much smaller the span of each sub-split must be compared to the span of the cluster being split.

use crate::{DistanceValue, NamedAlgorithm, utils::SizedHeap};

use super::{BipolarSplit, InitialPole, PartitionStrategy};

/// This strategy controls how much the span, and thus the radii, of child clusters must decrease compared to their parent cluster.
///
/// The `span` of a cluster is defined as the distance between any two of the extremal items in the cluster. This is analogous to, and indeed an approximation
/// of, the diameter of the cluster. The different variants of this strategy define a multiplicative factor by which the span of every sub-split must be smaller
/// than the span of the parent cluster. For example, if the factor is `2`, then the span of every sub-split must be no more than half that of the parent.
#[must_use]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub enum SpanReductionFactor {
    /// Use a fixed SRF value. This must be a finite value greater than `1.0`. If not, it will default to `√2`.
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
    GoldenRatio,
}

impl From<SpanReductionFactor> for PartitionStrategy {
    fn from(span_reduction: SpanReductionFactor) -> Self {
        Self::SpanReductionFactor(span_reduction)
    }
}

impl core::fmt::Display for SpanReductionFactor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Fixed(srf) => write!(f, "span-reduction-factor::fixed={srf}"),
            Self::Sqrt2 => write!(f, "span-reduction-factor::sqrt2"),
            Self::Two => write!(f, "span-reduction-factor::two"),
            Self::E => write!(f, "span-reduction-factor::e"),
            Self::Pi => write!(f, "span-reduction-factor::pi"),
            Self::GoldenRatio => write!(f, "span-reduction-factor::golden-ratio"),
        }
    }
}

impl core::str::FromStr for SpanReductionFactor {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let template = Self::regex_pattern();
        if let Some(caps) = template.captures(s) {
            if let Some(fixed) = caps.name("fixed") {
                let value = fixed.as_str().parse::<f64>().map_err(|e| format!("Failed to parse fixed SRF value: {e}"))?;
                Ok(Self::from(value))
            } else if s.ends_with("sqrt2") {
                Ok(Self::Sqrt2)
            } else if s.ends_with("two") {
                Ok(Self::Two)
            } else if s.ends_with('e') {
                Ok(Self::E)
            } else if s.ends_with("pi") {
                Ok(Self::Pi)
            } else if s.ends_with("golden-ratio") {
                Ok(Self::GoldenRatio)
            } else {
                Err(format!("Invalid SRF value: {s}"))
            }
        } else {
            Err(format!("Invalid SRF format: {s}"))
        }
    }
}

impl NamedAlgorithm for SpanReductionFactor {
    fn name(&self) -> &'static str {
        match self {
            Self::Fixed(_) => "span-reduction-factor::fixed",
            Self::Sqrt2 => "span-reduction-factor::sqrt2",
            Self::Two => "span-reduction-factor::two",
            Self::E => "span-reduction-factor::e",
            Self::Pi => "span-reduction-factor::pi",
            Self::GoldenRatio => "span-reduction-factor::golden-ratio",
        }
    }

    fn regex_pattern<'a>() -> &'a lazy_regex::Regex {
        lazy_regex::regex!(r"^span-reduction-factor::(fixed=(?P<fixed>[0-9]*\.?[0-9]+)|sqrt2|two|e|pi|golden-ratio)$")
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
        } else if (value - core::f64::consts::GOLDEN_RATIO).abs() < f64::EPSILON.sqrt() {
            Self::GoldenRatio
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
        } else if (value - core::f32::consts::GOLDEN_RATIO).abs() < f32::EPSILON.sqrt() {
            Self::GoldenRatio
        } else if 1.0 < value && value.is_finite() {
            Self::Fixed(f64::from(value))
        } else {
            Self::Sqrt2 // Default to Sqrt2 if out of range
        }
    }
}

impl SpanReductionFactor {
    /// Returns the maximum allowed child span for a given span from the parent cluster.
    fn max_child_span_for<T: DistanceValue>(self, parent_span: T) -> T {
        let factor = match self {
            Self::Fixed(srf) => srf,
            Self::Sqrt2 => core::f64::consts::SQRT_2,
            Self::Two => 2.0,
            Self::E => core::f64::consts::E,
            Self::Pi => core::f64::consts::PI,
            Self::GoldenRatio => core::f64::consts::GOLDEN_RATIO,
        };
        let parent_span = parent_span.to_f64().unwrap_or_else(|| unreachable!("DistanceValue must be convertible to f64"));
        T::from_f64(parent_span / factor).unwrap_or_else(|| unreachable!("DistanceValue must be convertible from f64"))
    }

    /// Splits the given items.
    pub(crate) fn split<'a, Id, I, T, M>(
        self,
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

        while splits.peek().is_some_and(|((items, _), (s, _))| items.len() > 1 && *s > max_span) {
            // Pop the widest split
            let ((items, distances), (_, ci)) = splits.pop().unwrap_or_else(|| unreachable!("child_items is not empty"));

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
        self,
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
        profi::prof!("SpanReductionFactor::par_split");

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

        while splits.peek().is_some_and(|((items, _), (s, _))| items.len() > 1 && *s > max_span) {
            // Pop the widest split
            let ((items, distances), (_, ci)) = splits.pop().unwrap_or_else(|| unreachable!("child_items is not empty"));

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
    // Behold the fancy heuristic!
    distances.iter().max_by_key(|&d| crate::utils::MaxItem((), *d)).map_or_else(T::zero, |&d| d)
}
