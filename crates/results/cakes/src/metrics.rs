//! Distance functions for use in the benchmarks.

use abd_clam::Metric;

use crate::AlignedSequence;

/// Distance functions for string data.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum StringDistance {
    /// `Levenshtein` distance.
    #[clap(name = "lev")]
    Levenshtein,
    /// `Needleman-Wunsch` distance.
    #[clap(name = "nw")]
    NeedlemanWunsch,
    /// `Hamming` distance.
    #[clap(name = "ham")]
    Hamming,
}

impl StringDistance {
    /// Get the distance function.
    #[allow(clippy::cast_possible_truncation)]
    pub fn metric(&self) -> Metric<AlignedSequence, u32> {
        let distance_function = match self {
            Self::Levenshtein => |x: &AlignedSequence, y: &AlignedSequence| {
                stringzilla::sz::edit_distance(x.as_unaligned(), y.as_unaligned()) as u32
            },
            Self::NeedlemanWunsch => |x: &AlignedSequence, y: &AlignedSequence| {
                distances::strings::nw_distance(&x.as_unaligned(), &y.as_unaligned())
            },
            Self::Hamming => {
                |x: &AlignedSequence, y: &AlignedSequence| distances::strings::hamming(x.sequence(), y.sequence())
            }
        };
        let expensive = !matches!(self, Self::Hamming);
        Metric::new(distance_function, expensive)
    }
}
