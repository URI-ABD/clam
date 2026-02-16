//! Search recall, the fraction of true neighbors that were found in the search results.

use crate::{
    DistanceValue, NamedAlgorithm,
    utils::{MeasurableQuality, MeasuredQuality, QualityMeasurer},
};

/// The Recall of a search algorithm.
///
/// It is defined as the fraction of true neighbors that were found in the search results.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct Recall;

impl_named_algorithm_for_unit_struct!(Recall, "recall", r"^recall$");

impl From<Recall> for super::MeasurableSearchQuality {
    fn from(_: Recall) -> Self {
        Self::Recall
    }
}

impl From<MeasuredQuality<Recall>> for super::SearchQuality {
    fn from(measured: MeasuredQuality<Recall>) -> Self {
        Self::Recall(measured)
    }
}

impl MeasurableQuality for Recall {
    fn is_higher_better(&self) -> bool {
        true
    }

    fn min_possible(&self) -> f64 {
        0.0
    }

    fn max_possible(&self) -> f64 {
        1.0
    }
}

/// `Recall` can be measured for search results, using the distance to the k-th nearest neighbor as a threshold for determining which predicted neighbors are
/// valid hits. This lets us break ties without having to rely the indices, or other identifiers, of the neighbors.
impl<T: DistanceValue> QualityMeasurer<(&[(usize, T)], T), usize> for Recall {
    #[expect(clippy::cast_precision_loss)]
    fn measure_once(&self, (search_results, k_th_distance): (&[(usize, T)], T), &k: &usize) -> f64 {
        let n_valid_hits = search_results.iter().filter(|&&(_, dist)| dist <= k_th_distance).count();
        n_valid_hits as f64 / k as f64
    }
}

/// `Recall` can also be measured for search results, using the true neighbors.
impl<T: DistanceValue> QualityMeasurer<(&[(usize, T)], &[(usize, T)]), ()> for Recall {
    fn measure_once(&self, (search_results, true_neighbors): (&[(usize, T)], &[(usize, T)]), (): &()) -> f64 {
        let kth_distance = max_distance_in(true_neighbors);
        let k = true_neighbors.len();
        self.measure_once((search_results, kth_distance), &k)
    }
}

/// Compute the maximum distance in a list of neighbors.
fn max_distance_in<T: DistanceValue>(neighbors: &[(usize, T)]) -> T {
    neighbors.iter().fold(T::min_value(), |max, &(_, dist)| if dist > max { dist } else { max })
}
