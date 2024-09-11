//! Distance functions for use in the benchmarks.

use abd_clam::Metric;
use distances::Number;

use crate::member_set::MemberSet;

// use crate::AlignedSequence;

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
    pub fn metric(&self) -> Metric<String, u32> {
        let distance_function = match self {
            Self::Levenshtein => |x: &String, y: &String| {
                // stringzilla::sz::edit_distance(x.as_unaligned(), y.as_unaligned()) as u32
                stringzilla::sz::edit_distance(x, y) as u32
            },
            Self::NeedlemanWunsch => |x: &String, y: &String| distances::strings::nw_distance(x, y),
            Self::Hamming => |x: &String, y: &String| distances::strings::hamming(x, y),
        };
        let expensive = !matches!(self, Self::Hamming);
        Metric::new(distance_function, expensive)
    }
}

/// Distance functions for set data.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum SetDistance {
    /// `Jaccard` distance.
    #[clap(name = "jac")]
    Jaccard,
}

impl SetDistance {
    /// Get the distance function.
    pub fn metric(&self) -> Metric<MemberSet, f32> {
        let distance_function = match self {
            Self::Jaccard => |x: &MemberSet, y: &MemberSet| {
                let intersection = x.inner().intersection(y.inner()).count();
                let union = x.len() + y.len() - intersection;
                let sim = if union == 0 {
                    0.0
                } else {
                    intersection.as_f32() / union.as_f32()
                };
                // mt_logger::mt_log!(mt_logger::Level::Debug, "Calculating Jaccard distance. x: {x:?}, y: {y:?}, intersection: {intersection}, union: {union}, sim: {sim}.");
                1.0 - sim
            },
        };
        Metric::new(distance_function, false)
    }
}
