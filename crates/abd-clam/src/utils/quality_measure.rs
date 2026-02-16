//! Various ways to measure the quality of an algorithm's output.

use rayon::prelude::*;

use super::NamedAlgorithm;

/// A trait for a quality measure that can be applied to the output of an algorithm.
pub trait MeasurableQuality: NamedAlgorithm {
    /// Returns whether this quality measure is better when it is higher or lower.
    fn is_higher_better(&self) -> bool;

    /// Returns the minimum possible value of this quality measure.
    fn min_possible(&self) -> f64;

    /// Returns the maximum possible value of this quality measure.
    fn max_possible(&self) -> f64;
}

/// A trait for methods that measure a `MeasurableQuality` and produce a `MeasuredQuality`.
pub trait QualityMeasurer<Args, BatchArgs>: MeasurableQuality {
    /// Measure the quality of a single output of an algorithm.
    fn measure_once(&self, args: Args, batch_args: &BatchArgs) -> f64;

    /// Measure the quality of a batch of outputs of an algorithm.
    ///
    /// The default implementation simply applies `measure_once` to each item in the batch and collects the results.
    fn measure_batch<I>(self, iter_args: I, batch_args: &BatchArgs) -> MeasuredQuality<Self>
    where
        I: Iterator<Item = Args>,
    {
        let observed_values = iter_args.map(|arg| self.measure_once(arg, batch_args)).collect::<Vec<_>>();
        MeasuredQuality::from((self, observed_values))
    }

    /// Parallel version of [`Self::measure_batch`].
    fn par_measure_batch<I>(self, par_iter_args: I, batch_args: &BatchArgs) -> MeasuredQuality<Self>
    where
        Self: Sync,
        Args: Sync,
        BatchArgs: Sync,
        I: ParallelIterator<Item = Args>,
    {
        let observed_values = par_iter_args.map(|arg| self.measure_once(arg, batch_args)).collect::<Vec<_>>();
        MeasuredQuality::from((self, observed_values))
    }
}

/// The result of applying a quality measure to the output of an algorithm.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[derive(Debug, Clone)]
#[must_use]
pub struct MeasuredQuality<Q: MeasurableQuality> {
    /// The quality measure that was applied.
    measure: Q,
    /// The minimum observed value of the quality measure.
    min: f64,
    /// The maximum observed value of the quality measure.
    max: f64,
    /// The mean of the observed values of the quality measure.
    mean: f64,
    /// The standard deviation of the observed values of the quality measure.
    std_dev: f64,
    /// All observed values of the quality measure.
    observed_values: Vec<f64>,
}

impl<Q: MeasurableQuality> core::fmt::Display for MeasuredQuality<Q> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let higher_better = if self.is_higher_better() { "higher" } else { "lower" };
        write!(
            f,
            "{}(mean={:.4}, std_dev={:.4}, min={:.4}, max={:.4}), {higher_better} is better",
            self.measure.name(),
            self.mean(),
            self.std_dev(),
            self.min(),
            self.max(),
        )
    }
}

impl<Q: MeasurableQuality> From<(Q, Vec<f64>)> for MeasuredQuality<Q> {
    fn from((measure, observed_values): (Q, Vec<f64>)) -> Self {
        #[expect(clippy::cast_precision_loss)]
        let batch_size = observed_values.len() as f64;
        let min = observed_values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = observed_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = observed_values.iter().sum::<f64>() / batch_size;
        let std_dev = (observed_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / batch_size).sqrt();
        Self {
            measure,
            min,
            max,
            mean,
            std_dev,
            observed_values,
        }
    }
}

impl<Q: MeasurableQuality> MeasuredQuality<Q> {
    /// Aggregates many measurements of the same quality measure into a single `MeasuredQuality` that summarizes the results.
    ///
    /// This is useful for aggregating the results of multiple batches of search queries into a single `MeasuredQuality` that summarizes the overall quality of
    /// the search algorithm across all batches. The aggregation will combine all observed values from the individual measurements and compute the overall min,
    /// max, mean, and standard deviation of the observed values.
    ///
    /// # Errors
    ///
    /// - If the list of measurements is empty.
    pub fn aggregate(mut measurements: Vec<Self>) -> Result<Self, String> {
        if measurements.is_empty() {
            return Err("Cannot aggregate an empty list of measurements".to_string());
        }

        let observations = measurements
            .iter_mut()
            .flat_map(|m| core::mem::take(&mut m.observed_values))
            .collect::<Vec<_>>();
        let measure = measurements
            .pop()
            .unwrap_or_else(|| unreachable!("We just checked that measurements is not empty, so pop should never return None"))
            .measure;

        Ok(Self::from((measure, observations)))
    }

    /// Returns the measured quality of the algorithm's output.
    pub const fn measure(&self) -> &Q {
        &self.measure
    }

    /// Returns whether this quality measure is better when it is higher or lower.
    pub fn is_higher_better(&self) -> bool {
        self.measure.is_higher_better()
    }

    /// Returns the minimum possible value of the quality measure.
    pub fn min_possible(&self) -> f64 {
        self.measure.min_possible()
    }

    /// Returns the maximum possible value of the quality measure.
    pub fn max_possible(&self) -> f64 {
        self.measure.max_possible()
    }

    /// Returns the minimum observed value of the quality measure.
    pub const fn min(&self) -> f64 {
        self.min
    }

    /// Returns the maximum observed value of the quality measure.
    pub const fn max(&self) -> f64 {
        self.max
    }

    /// Returns the mean of the observed values of the quality measure.
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the standard deviation of the observed values of the quality measure.
    pub const fn std_dev(&self) -> f64 {
        self.std_dev
    }

    /// Returns all observed values of the quality measure.
    pub fn observed_values(&self) -> &[f64] {
        &self.observed_values
    }
}
