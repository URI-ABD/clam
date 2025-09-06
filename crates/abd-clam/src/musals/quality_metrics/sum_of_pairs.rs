//! Scores each pairwise alignment in the MSA, applying a penalty for gaps and mismatches.

use rayon::prelude::*;

use crate::{DistanceValue, musals::Sequence};

use super::{MsaQuality, mu_sigma_min_max, random_sample_indices};

/// The Sum of Pairs (SP) score of the MSA.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct SumOfPairs {
    /// Mean
    mean: f64,
    /// Standard Deviation
    std_dev: f64,
    /// Minimum
    min: f64,
    /// Maximum
    max: f64,
}

impl MsaQuality for SumOfPairs {
    fn name(&self) -> String {
        "SumOfPairs".to_string()
    }

    fn short_name<'a>(&self) -> &'a str {
        "sp"
    }

    fn description(&self) -> String {
        "The Sum of Pairs (SP) score of the MSA.".to_string()
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn std_dev(&self) -> f64 {
        self.std_dev
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }

    fn compute<Id, S, T, M>(aligned_items: &[(Id, S)], _: &M, sample_size: Option<usize>) -> Self
    where
        S: Sequence,
        T: DistanceValue,
        M: Fn(&S, &S) -> T,
        Self: Sized,
    {
        let indices = random_sample_indices(aligned_items.len(), sample_size);
        let sequences = indices.iter().map(|&i| &aligned_items[i].1).collect::<Vec<_>>();
        let columns = msa_to_columns(&sequences);
        let scores = columns.iter().map(column_score).collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(&scores);
        Self { mean, std_dev, min, max }
    }

    fn par_compute<Id, S, T, M>(aligned_items: &[(Id, S)], _: &M, sample_size: Option<usize>) -> Self
    where
        Id: Send + Sync,
        S: Sequence + Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&S, &S) -> T + Send + Sync,
        Self: Sized + Send + Sync,
    {
        let indices = random_sample_indices(aligned_items.len(), sample_size);
        let sequences = indices.iter().map(|&i| &aligned_items[i].1).collect::<Vec<_>>();
        let columns = par_msa_to_columns(&sequences);
        let scores = columns.par_iter().map(column_score).collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(&scores);
        Self { mean, std_dev, min, max }
    }
}

/// Converts the MSA from rows to columns.
fn msa_to_columns<S>(sequences: &[&S]) -> Vec<S>
where
    S: Sequence,
{
    let sequences = sequences.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    let n_cols = sequences[0].len();
    (0..n_cols)
        .map(|col_idx| S::from_vec(sequences.iter().map(|seq| seq[col_idx]).collect()))
        .collect()
}

/// Parallel version of [`msa_to_columns`].
fn par_msa_to_columns<S>(sequences: &[&S]) -> Vec<S>
where
    S: Sequence + Send + Sync,
{
    let sequences = sequences.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    let n_cols = sequences[0].len();
    (0..n_cols)
        .into_par_iter()
        .map(|col_idx| S::from_vec(sequences.iter().map(|seq| seq[col_idx]).collect()))
        .collect()
}

/// Computes the Sum of Pairs score for a single column in the MSA, normalized by the number of pairs.
fn column_score<S>(column: &S) -> f64
where
    S: Sequence,
{
    let frequencies = count_pairs(column.as_ref());
    let total_pairs: usize = frequencies.iter().sum();

    let score = frequencies
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .inspect(|(i, _)| ftlog::debug!("Calculating SoP contribution of pair index {i}"))
        .fold(0_f64, |acc, (k, &count)| {
            let (i, j) = flat_to_lt_index(k);
            let pair_score = if i == j { 0.0 } else { 1.0 };
            (count as f64).mul_add(pair_score, acc)
        });

    score / (total_pairs as f64)
}

/// Counts the frequency of each pair of different characters in the column.
fn count_pairs(column: &[u8]) -> Vec<usize> {
    let mut pair_counts = vec![0; 255 * 128]; // Assuming ASCII characters

    for (i, &a) in column.iter().enumerate() {
        for &b in &column[i + 1..] {
            let (i, j) = if a < b { (a, b) } else { (b, a) };
            pair_counts[lt_to_flat_index(i as usize, j as usize)] += 1;
        }
    }

    pair_counts
}

/// Converts a pair of indices for a lower-triangular matrix into a single index in a flat array.
///
/// This assumes that `i >= j`.
const fn lt_to_flat_index(i: usize, j: usize) -> usize {
    (i * (i + 1)) / 2 + j
}

/// Converts a flat array index into a pair of indices for a lower-triangular matrix.
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::many_single_char_names)]
fn flat_to_lt_index(k: usize) -> (usize, usize) {
    let p = ((1 + 8 * k) as f64).sqrt();
    let i = ((p - 1.0) / 2.0).floor() as usize;

    let t = i * (i + 1) / 2;
    let j = k - t;

    (i, j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lt_to_flat_index() {
        let expected_pairs = [
            ((0, 0), 0), // (0,0) -> 0
            ((1, 0), 1), // (1,0) -> 1
            ((1, 1), 2), // (1,1) -> 2
            ((2, 0), 3), // (2,0) -> 3
            ((2, 1), 4), // (2,1) -> 4
            ((2, 2), 5), // (2,2) -> 5
            ((3, 0), 6), // (3,0) -> 6
            ((3, 1), 7), // (3,1) -> 7
            ((3, 2), 8), // (3,2) -> 8
            ((3, 3), 9), // (3,3) -> 9
        ];
        for ((i, j), k) in expected_pairs {
            assert_eq!(lt_to_flat_index(i, j), k);
            assert_eq!((i, j), flat_to_lt_index(k));
        }

        let mut k = 0;
        for i in 0..=255 {
            for j in 0..=i {
                assert_eq!(lt_to_flat_index(i, j), k);

                let (a, b) = flat_to_lt_index(k);
                assert_eq!((i, j), (a, b), "Failed at k = {k}");

                k += 1;
            }
        }
    }
}
