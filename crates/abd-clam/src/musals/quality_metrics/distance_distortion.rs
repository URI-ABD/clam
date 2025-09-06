//! Distance Distortion MSA Quality Metric

use rayon::prelude::*;

use crate::{DistanceValue, musals::Sequence};

use super::{MsaQuality, mu_sigma_min_max, random_sample_indices};

/// The scores of pairwise alignments in the MSA.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct DistanceDistortion {
    /// Mean
    mean: f64,
    /// Standard Deviation
    std_dev: f64,
    /// Minimum
    min: f64,
    /// Maximum
    max: f64,
}

impl MsaQuality for DistanceDistortion {
    fn name(&self) -> String {
        "DistanceDistortion".to_string()
    }

    fn short_name<'a>(&self) -> &'a str {
        "dd"
    }

    fn description(&self) -> String {
        "The mean distortion of alignment distances between pairs of sequences in the MSA.".to_string()
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

    fn compute<Id, S, T, M>(aligned_items: &[(Id, S)], metric: &M, sample_size: Option<usize>) -> Self
    where
        S: Sequence,
        T: DistanceValue,
        M: Fn(&S, &S) -> T,
        Self: Sized,
    {
        let indices = random_sample_indices(aligned_items.len(), sample_size);
        let sequences = indices.iter().map(|&i| &aligned_items[i].1).collect::<Vec<_>>();

        let pairwise_scores = sequences
            .iter()
            .enumerate()
            .flat_map(|(i, &s1)| {
                sequences
                    .iter()
                    .enumerate()
                    .skip(i + 1)
                    .inspect(move |(j, _)| ftlog::debug!("Calculating distance distortion for sequence pair ({i}, {j})"))
                    .map(move |(_, &s2)| (s1, s2))
            })
            .map(|(s1, s2)| dd_inner(s1, s2, metric))
            .collect::<Vec<_>>();

        let (mean, std_dev, min, max) = mu_sigma_min_max(&pairwise_scores);
        Self { mean, std_dev, min, max }
    }

    fn par_compute<Id, S, T, M>(aligned_items: &[(Id, S)], metric: &M, sample_size: Option<usize>) -> Self
    where
        Id: Send + Sync,
        S: Sequence + Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&S, &S) -> T + Send + Sync,
        Self: Sized + Send + Sync,
    {
        let indices = random_sample_indices(aligned_items.len(), sample_size);
        let sequences = indices.iter().map(|&i| &aligned_items[i].1).collect::<Vec<_>>();

        let pairwise_scores = sequences
            .par_iter()
            .enumerate()
            .flat_map(|(i, &s1)| {
                sequences
                    .par_iter()
                    .enumerate()
                    .skip(i + 1)
                    .inspect(move |(j, _)| ftlog::debug!("Calculating distance distortion for sequence pair ({i}, {j})"))
                    .map(move |(_, &s2)| (s1, s2))
            })
            .map(|(s1, s2)| dd_inner(s1, s2, metric))
            .collect::<Vec<_>>();

        let (mean, std_dev, min, max) = mu_sigma_min_max(&pairwise_scores);
        Self { mean, std_dev, min, max }
    }
}

/// Measures the distortion of the Levenshtein edit distance between the unaligned sequences and the Hamming distance between the aligned sequences.
fn dd_inner<S: Sequence, T: DistanceValue, M: Fn(&S, &S) -> T>(s1: &S, s2: &S, metric: &M) -> f64 {
    let ham = s1.as_ref().iter().zip(s2.as_ref().iter()).filter(|(a, b)| a != b).count();
    let met = metric(&s1.without_gaps(), &s2.without_gaps());
    if met == T::zero() {
        1.0
    } else {
        let ham = ham as f64;
        let met = met.to_f64().unwrap_or_else(|| unreachable!("DistanceValue conversion to f64 failed"));
        ham / met
    }
}
