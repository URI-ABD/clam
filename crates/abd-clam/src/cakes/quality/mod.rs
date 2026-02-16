//! Measuring search quality using recall, relative distance error, and other metrics.

use rayon::prelude::*;

use crate::{
    DistanceValue, NamedAlgorithm,
    utils::{MeasuredQuality, QualityMeasurer},
};

mod recall;
mod relative_distance_error;

pub use recall::Recall;
pub use relative_distance_error::RelativeDistanceError;

/// Quality metrics for search algorithms that can be measured and reported.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[cfg_attr(feature = "shell", derive(clap::ValueEnum))]
pub enum MeasurableSearchQuality {
    /// Recall, the fraction of true neighbors that were found in the search results.
    #[cfg_attr(feature = "shell", clap(name = "recall"))]
    Recall,
    /// Relative distance error, the average relative error of the distances of the search results compared to the true neighbors.
    #[cfg_attr(feature = "shell", clap(name = "relative-distance-error"))]
    RelativeDistanceError,
}

/// An enum to account for the types of search quality metrics that have been measured.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[must_use]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub enum SearchQuality {
    /// Recall, the fraction of true neighbors that were found in the search results.
    Recall(MeasuredQuality<Recall>),
    /// Relative distance error, the average relative error of the distances of the search results compared to the true neighbors.
    RelativeDistanceError(MeasuredQuality<RelativeDistanceError>),
}

impl core::fmt::Display for SearchQuality {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Recall(recall) => recall.fmt(f),
            Self::RelativeDistanceError(rde) => rde.fmt(f),
        }
    }
}

impl SearchQuality {
    /// Aggregate a batch of search quality measurements into a single measurement, if applicable.
    ///
    /// This is useful for aggregating the results of multiple batches of search queries into a single measurement for each quality metric.
    ///
    /// # Arguments
    ///
    /// - `measurements`: A vector of search quality measurements to aggregate.
    ///
    /// # Returns
    ///
    /// A tuple of two optional aggregated search quality measurements, one for each quality metric. If a quality metric cannot be aggregated (e.g., if no
    /// batches were measured for that metric), the corresponding entry in the tuple will be `None`.
    #[must_use]
    pub fn aggregate_batch(measurements: Vec<Self>) -> (Option<MeasuredQuality<Recall>>, Option<MeasuredQuality<RelativeDistanceError>>) {
        let (recalls, rdes) = measurements.into_iter().fold((Vec::new(), Vec::new()), |(mut recalls, mut rdes), measurement| {
            match measurement {
                Self::Recall(recall) => recalls.push(recall),
                Self::RelativeDistanceError(rde) => rdes.push(rde),
            }
            (recalls, rdes)
        });
        let aggregated_recalls = MeasuredQuality::aggregate(recalls).ok();
        let aggregated_rdes = MeasuredQuality::aggregate(rdes).ok();
        (aggregated_recalls, aggregated_rdes)
    }
}

impl core::fmt::Display for MeasurableSearchQuality {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Recall => Recall.fmt(f),
            Self::RelativeDistanceError => RelativeDistanceError.fmt(f),
        }
    }
}

impl core::str::FromStr for MeasurableSearchQuality {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::regex_pattern().captures(s).map_or_else(
            || Err(format!("Invalid format for MeasurableSearchQuality: {s}")),
            |caps| {
                let metric = caps.get(1).map(|m| m.as_str());
                match metric {
                    Some("recall") => Recall::from_str(s).map(Self::from),
                    Some("relative-distance-error") => RelativeDistanceError::from_str(s).map(Self::from),
                    Some(metric) => Err(format!("Unknown search quality metric: {metric}")),
                    None => Err(format!("Invalid format for MeasurableSearchQuality: {s}")),
                }
            },
        )
    }
}

impl NamedAlgorithm for MeasurableSearchQuality {
    fn name(&self) -> &'static str {
        match self {
            Self::Recall => Recall.name(),
            Self::RelativeDistanceError => RelativeDistanceError.name(),
        }
    }

    fn regex_pattern<'a>() -> &'a lazy_regex::Regex {
        lazy_regex::regex!(r"^(recall|relative-distance-error)$")
    }
}

impl MeasurableSearchQuality {
    /// Return all variants of `MeasurableSearchQuality` as a vector.
    #[must_use]
    pub fn all_variants() -> Vec<Self> {
        vec![Self::Recall, Self::RelativeDistanceError]
    }

    /// Measures the quality of a batch of search results against the true neighbors.
    pub fn measure_batch<T: DistanceValue>(&self, search_results: &[Vec<(usize, T)>], true_neighbors: &[Vec<(usize, T)>]) -> SearchQuality {
        let paired_results = search_results.iter().zip(true_neighbors).map(|(a, b)| (a.as_slice(), b.as_slice()));
        match self {
            Self::Recall => SearchQuality::from(Recall.measure_batch(paired_results, &())),
            Self::RelativeDistanceError => SearchQuality::from(RelativeDistanceError.measure_batch(paired_results, &())),
        }
    }

    /// Parallel version of [`Self::measure_batch`].
    pub fn par_measure_batch<T: DistanceValue + Send + Sync>(&self, search_results: &[Vec<(usize, T)>], true_neighbors: &[Vec<(usize, T)>]) -> SearchQuality {
        let paired_results = search_results
            .par_iter()
            .zip(true_neighbors.par_iter())
            .map(|(a, b)| (a.as_slice(), b.as_slice()));
        match self {
            Self::Recall => SearchQuality::from(Recall.par_measure_batch(paired_results, &())),
            Self::RelativeDistanceError => SearchQuality::from(RelativeDistanceError.par_measure_batch(paired_results, &())),
        }
    }
}
