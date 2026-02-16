//! MSA quality measure: sum of pairs

use crate::{
    DistanceValue, NamedAlgorithm,
    utils::{MeasurableQuality, MeasuredQuality, QualityMeasurer},
};

use super::CostMatrix;

/// Sum-of-pairs score, which measures the fraction of mismatched aligned residue pairs in the alignment.
#[derive(Debug, Clone)]
#[must_use]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct SumOfPairs;

impl_named_algorithm_for_unit_struct!(SumOfPairs, "sum-of-pairs", r"^sum-of-pairs$");

impl From<SumOfPairs> for super::MeasurableAlignmentQuality {
    fn from(_: SumOfPairs) -> Self {
        Self::SumOfPairs
    }
}

impl From<MeasuredQuality<SumOfPairs>> for super::AlignmentQuality {
    fn from(measured: MeasuredQuality<SumOfPairs>) -> Self {
        Self::SumOfPairs(measured)
    }
}

impl MeasurableQuality for SumOfPairs {
    fn is_higher_better(&self) -> bool {
        false
    }

    fn min_possible(&self) -> f64 {
        0.0
    }

    fn max_possible(&self) -> f64 {
        1.0
    }
}

impl<T: DistanceValue> QualityMeasurer<&[char], CostMatrix<T>> for SumOfPairs {
    #[expect(clippy::cast_precision_loss)]
    fn measure_once(&self, col: &[char], cost_matrix: &CostMatrix<T>) -> f64 {
        let n = col.len();
        if n < 2 {
            return 1.0; // If there's only one sequence, we consider it perfectly aligned
        }
        let n_pairs = (n * (n - 1)) / 2;
        let frequencies = pair_frequencies(col, n_pairs);
        let col_score = frequencies
            .into_iter()
            .map(|((res1, res2), freq)| {
                let pair_score = cost_matrix
                    .sub_cost(res1, res2)
                    .to_f64()
                    .unwrap_or_else(|| unreachable!("DistanceValue must be convertible to f64"));
                pair_score * (freq as f64)
            })
            .sum::<f64>();
        col_score / (n_pairs as f64)
    }
}

/// Computes the frequency of each residue pair in a given column of the MSA.
fn pair_frequencies(column: &[char], n_pairs: usize) -> Vec<((char, char), usize)> {
    let mut freq_map = std::collections::HashMap::with_capacity(n_pairs);
    for (i, &res1) in column.iter().enumerate() {
        for &res2 in &column[(i + 1)..] {
            *freq_map.entry((res1, res2)).or_insert(0) += 1;
        }
    }
    freq_map.into_iter().collect()
}
