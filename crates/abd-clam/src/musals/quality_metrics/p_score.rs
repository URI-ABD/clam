//! The fraction of mismatches between pairs of sequences in a multiple sequence alignment (MSA).

use rayon::prelude::*;

use crate::{DistanceValue, musals::Sequence};

use super::{MsaQuality, mu_sigma_min_max, random_sample_indices};

/// The fraction of mismatches between unit pairs among sequences in the MSA.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct PScore {
    /// Mean
    mean: f64,
    /// Standard Deviation
    std_dev: f64,
    /// Minimum
    min: f64,
    /// Maximum
    max: f64,
}

impl MsaQuality for PScore {
    fn name(&self) -> String {
        "PScore".to_string()
    }

    fn short_name<'a>(&self) -> &'a str {
        "ps"
    }

    fn description(&self) -> String {
        "The mean fraction of mismatched unit pairs between all pairs of sequences in the MSA.".to_string()
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

        let (mean, std_dev, min, max) = mu_sigma_min_max(ps_inner(&sequences));
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

        let (mean, std_dev, min, max) = mu_sigma_min_max(par_ps_inner(&sequences));
        Self { mean, std_dev, min, max }
    }
}

/// A helper for calculating mismatch fractions between all pairs of sequences.
fn ps_inner<S: Sequence>(sequences: &[&S]) -> Vec<f64> {
    sequences
        .iter()
        .enumerate()
        .flat_map(|(i, s1)| {
            sequences
                .iter()
                .enumerate()
                .skip(i + 1)
                .inspect(move |(j, _)| ftlog::debug!("Calculating mismatch fraction for sequence pair ({i}, {j})"))
                .map(move |(_, s2)| ps_single(s1.as_ref(), s2.as_ref(), S::GAP))
        })
        .collect()
}

/// A helper for calculating mismatch fraction of a single pair of sequences.
fn ps_single(s1: &[u8], s2: &[u8], gap: u8) -> f64 {
    let num_mismatches = s1.iter().zip(s2.iter()).filter(|&(&a, &b)| a != gap && b != gap && a != b).count();
    num_mismatches as f64 / s1.len() as f64
}

/// Parallel version of [`ps_inner`].
fn par_ps_inner<S: Sequence + Send + Sync>(sequences: &[&S]) -> Vec<f64> {
    sequences
        .par_iter()
        .enumerate()
        .flat_map(|(i, s1)| {
            sequences
                .par_iter()
                .enumerate()
                .skip(i + 1)
                .inspect(move |(j, _)| ftlog::debug!("Calculating mismatch fraction for sequence pair ({i}, {j})"))
                .map(move |(_, s2)| ps_single(s1.as_ref(), s2.as_ref(), S::GAP))
        })
        .collect()
}
