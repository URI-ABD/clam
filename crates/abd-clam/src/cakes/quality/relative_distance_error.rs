//! Relative Distance Error (RDE), a measure of search quality that accounts for the relative distances of the search results compared to the ground truth.

use crate::{
    DistanceValue, NamedAlgorithm,
    utils::{MeasurableQuality, MeasuredQuality, QualityMeasurer},
};

/// Relative Distance Error (RDE).
///
/// Given a set of search results and the corresponding ground truth, RDE is calculated as follows:
///
/// 1. Sort the search results and the ground truth by distance in non-descending order.
/// 2. For each pair of corresponding results (one from the search results and one from the ground truth), calculate the relative distance error as
///    `(d_pred / d_true) - 1`, where `d_pred` is the distance of the search result and `d_true` is the distance of the ground truth result.
/// 3. Compute the mean of the RDEs for all pairs of results.
///
/// If the ground truth is empty, RDE is defined as `0` if the search results are also empty, and `f64::INFINITY` otherwise. If the number of search results
/// does not match the number of ground truth results, RDE is defined as `f64::INFINITY`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct RelativeDistanceError;

impl_named_algorithm_for_unit_struct!(RelativeDistanceError, "relative-distance-error", r"^relative-distance-error$");

impl From<RelativeDistanceError> for super::MeasurableSearchQuality {
    fn from(_: RelativeDistanceError) -> Self {
        Self::RelativeDistanceError
    }
}

impl From<MeasuredQuality<RelativeDistanceError>> for super::SearchQuality {
    fn from(measured: MeasuredQuality<RelativeDistanceError>) -> Self {
        Self::RelativeDistanceError(measured)
    }
}

impl MeasurableQuality for RelativeDistanceError {
    fn is_higher_better(&self) -> bool {
        false
    }

    fn min_possible(&self) -> f64 {
        0.0
    }

    fn max_possible(&self) -> f64 {
        // TODO: Find a practical upper bound for RDE, to avoid issues when averaging large numbers of measurements containing some infinite values.
        f64::INFINITY
    }
}

impl<T: DistanceValue> QualityMeasurer<(&[(usize, T)], &[(usize, T)]), ()> for RelativeDistanceError {
    fn measure_once(&self, (search_results, true_neighbors): (&[(usize, T)], &[(usize, T)]), (): &()) -> f64 {
        if true_neighbors.is_empty() {
            if search_results.is_empty() {
                self.min_possible()
            } else {
                self.max_possible()
            }
        } else if search_results.len() == true_neighbors.len() {
            #[expect(clippy::cast_precision_loss)]
            let n_hits = true_neighbors.len() as f64;

            let results = sorted_by_distance(search_results);
            let ground_truth = sorted_by_distance(true_neighbors);

            let err_sum = ground_truth
                .iter()
                .zip(results)
                .map(|((_, d_true), (_, d_pred))| {
                    let d_true = d_true
                        .to_f64()
                        .unwrap_or_else(|| unreachable!("Distance value should always be convertible to f64"));
                    let d_pred = d_pred
                        .to_f64()
                        .unwrap_or_else(|| unreachable!("Distance value should always be convertible to f64"));
                    if d_true == 0.0 || d_pred == 0.0 { 0.0 } else { d_pred / d_true - 1.0 }
                })
                .sum::<f64>();

            err_sum / n_hits
        } else {
            self.max_possible()
        }
    }
}

/// Sorts search results by distance in non-descending order.
fn sorted_by_distance<T: DistanceValue>(results: &[(usize, T)]) -> Vec<(usize, T)> {
    let mut results = results.to_vec();
    results.sort_by_key(|&(_, d)| crate::utils::MinItem((), d));
    results
}
