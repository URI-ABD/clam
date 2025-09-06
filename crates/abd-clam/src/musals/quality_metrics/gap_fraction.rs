//! The mean fraction of gaps in the sequences of a multiple sequence alignment (MSA).

use rayon::prelude::*;

use crate::{
    DistanceValue,
    musals::{Sequence, quality_metrics::random_sample_indices},
};

use super::{MsaQuality, mu_sigma_min_max};

/// The mean of the fraction of gaps in the sequences of the MSA.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct GapFraction {
    /// Mean
    mean: f64,
    /// Standard Deviation
    std_dev: f64,
    /// Minimum
    min: f64,
    /// Maximum
    max: f64,
}

impl MsaQuality for GapFraction {
    fn name(&self) -> String {
        "MeanGapFraction".to_string()
    }

    fn short_name<'a>(&self) -> &'a str {
        "gf"
    }

    fn description(&self) -> String {
        "The mean of the fraction of gaps in each sequence of the MSA.".to_string()
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
        let msa_width = aligned_items[0].1.as_ref().len() as f64;

        let indices = random_sample_indices(aligned_items.len(), sample_size);
        let gap_fractions = indices
            .iter()
            .map(|&i| aligned_items[i].1.gap_count())
            .map(|c| c as f64 / msa_width)
            .collect::<Vec<_>>();

        let (mean, std_dev, min, max) = mu_sigma_min_max(gap_fractions);
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
        let msa_width = aligned_items[0].1.as_ref().len() as f64;

        let indices = random_sample_indices(aligned_items.len(), sample_size);
        let gap_fractions = indices
            .par_iter()
            .map(|&i| aligned_items[i].1.gap_count())
            .map(|c| c as f64 / msa_width)
            .collect::<Vec<_>>();

        let (mean, std_dev, min, max) = mu_sigma_min_max(gap_fractions);
        Self { mean, std_dev, min, max }
    }
}
