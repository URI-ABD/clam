//! Metrics for use with the CAKES datasets

use distances::{
    Number,
    number::{Float, Int},
};

/// The available metrics for CAKES datasets.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum Metric {
    #[clap(name = "lcs")]
    Lcs,
    #[clap(name = "levenshtein")]
    Levenshtein,
    #[clap(name = "euclidean")]
    Euclidean,
    #[clap(name = "cosine")]
    Cosine,
}

impl Metric {
    /// Get the `Metric` for the distance function.
    pub fn name(&self) -> &'static str {
        match self {
            Metric::Lcs => "lcs",
            Metric::Levenshtein => "levenshtein",
            Metric::Euclidean => "euclidean",
            Metric::Cosine => "cosine",
        }
    }
}

pub fn lcs<I: AsRef<[u8]>, T: Int>(a: &I, b: &I) -> T {
    let (a, b) = (a.as_ref(), b.as_ref());

    let mut dp = vec![vec![T::ZERO; b.len() + 1]; a.len() + 1];
    for (i, &a_byte) in a.iter().enumerate() {
        for (j, &b_byte) in b.iter().enumerate() {
            if a_byte == b_byte {
                dp[i + 1][j + 1] = dp[i][j] + T::ONE;
            } else {
                dp[i + 1][j + 1] = core::cmp::Ord::max(dp[i + 1][j], dp[i][j + 1]);
            }
        }
    }

    dp[a.len()][b.len()]
}

pub fn levenshtein<I: AsRef<[u8]>, T: Int>(a: &I, b: &I) -> T {
    let sz_device = stringzilla::szs::DeviceScope::default().unwrap();
    let sz_engine = stringzilla::szs::LevenshteinDistances::new(&sz_device, 0, 1, 1, 1).unwrap();
    let distances = sz_engine.compute(&sz_device, &[a], &[b]).unwrap();
    T::from(distances[0])
}

pub fn euclidean<I: AsRef<[T]>, T: Number, U: Float>(a: &I, b: &I) -> U {
    distances::vectors::euclidean(a.as_ref(), b.as_ref())
}

pub fn cosine<I: AsRef<[T]>, T: Number, U: Float>(a: &I, b: &I) -> U {
    distances::vectors::cosine(a.as_ref(), b.as_ref())
}
