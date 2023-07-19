//! String distance metrics.

// use alloc::vec::Vec;  // no-std

use crate::number::UInt;

pub mod needleman_wunsch;

pub use needleman_wunsch::nw_distance;

/// Penalties to use in the Needleman-Wunsch distance calculation.
///
/// Since we provide a distance implementation that is intended to be used as a
/// metric that obeys the triangle inequality, the penalties should all be
/// non-negative, thus the genericity over unsigned integers.
#[derive(Clone, Copy, Debug)]
pub struct Penalties<U: UInt> {
    /// Penalty for a match.
    pub(crate) match_: U,
    /// Penalty for a mis-match.
    pub(crate) mismatch: U,
    /// Penalty for a gap.
    pub(crate) gap: U,
}

impl<U: UInt> Default for Penalties<U> {
    fn default() -> Self {
        Self {
            match_: U::zero(),
            mismatch: U::one(),
            gap: U::one(),
        }
    }
}

impl<U: UInt> Penalties<U> {
    /// Create a set of penalties to use for the NW distance metric.
    pub const fn new(match_: U, mismatch: U, gap: U) -> Self {
        Self {
            match_,
            mismatch,
            gap,
        }
    }
}

/// Creates a function to compute the Levenshtein distance between two strings
/// using a custom set of penalties. The generated function will have the same
/// signature as `levenshtein`.
///
/// # Arguments
///
/// * `penalties`: the set of penalties to use
pub fn levenshtein_custom<U: UInt>(penalties: Penalties<U>) -> impl Fn(&str, &str) -> U {
    move |x: &str, y: &str| {
        U::from(if x.is_empty() {
            // handle special case of 0 length
            y.len()
        } else if y.is_empty() {
            // handle special case of 0 length
            x.len()
        } else if x.len() < y.len() {
            // require tat a is no shorter than b
            _levenshtein(y, x, penalties)
        } else {
            _levenshtein(x, y, penalties)
        })
    }
}

/// Computes the Levenshtein distance between two strings.
///
/// The Levenshtein distance is defined as the minimum number of edits
/// needed to transform one string into the other, with the allowable
/// edit operations being insertion, deletion, or substitution of a
/// single character. It is named after Vladimir Levenshtein, who
/// considered this distance in 1965.
///
/// We use the Wagner-Fischer algorithm to compute the Levenshtein
/// distance. The Wagner-Fischer algorithm is a dynamic programming
/// algorithm that computes the edit distance between two strings of
/// characters.
///
/// We use penalty values of `1` for all edit operations and we minimize the
/// total penalty for aligning the two strings.
///
/// The input strings are not required to be of the same length.
///
/// # Arguments
///
/// * `x`: The first string.
/// * `y`: The second string.
///
/// # Examples
///
/// ```
/// use distances::strings::levenshtein;
///
/// let x = "NAJIBEATSPEPPERS";
/// let y = "NAJIBPEPPERSEATS";
///
/// let distance: u16 = levenshtein(x, y);
///
/// assert_eq!(distance, 8);
///
/// let x = "TOMEATSWHATFOODEATS";
/// let y = "FOODEATSWHATTOMEATS";
///
/// let distance: u16 = levenshtein(x, y);
///
/// assert_eq!(distance, 6);
/// ```
///
/// # References
///
/// * [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
#[must_use]
pub fn levenshtein<U: UInt>(x: &str, y: &str) -> U {
    U::from(if x.is_empty() {
        // handle special case of 0 length
        y.len()
    } else if y.is_empty() {
        // handle special case of 0 length
        x.len()
    } else if x.len() < y.len() {
        // require tat a is no shorter than b
        _levenshtein(y, x, Penalties::<U>::default())
    } else {
        _levenshtein(x, y, Penalties::<U>::default())
    })
}

/// Helper for Levenshtein distance.
#[allow(unused_variables)]
fn _levenshtein<U: UInt>(x: &str, y: &str, penalties: Penalties<U>) -> usize {
    // TODO(Noah): Please use the penalties struct. Modify the members if you
    // need to. Keep in mind that this struct is also used for Needleman-
    // Wunsch and will be used for the other distance metrics we implement.

    // initialize DP table for string b
    let mut cur = (0..=y.len()).collect::<Vec<_>>();

    // calculate edit distance
    for (i, c_x) in x.chars().enumerate() {
        // get first column for this row
        let mut pre = cur[0];
        cur[0] = i + 1;
        for (j, c_y) in y.chars().enumerate() {
            let tmp = cur[j + 1];
            cur[j + 1] = core::cmp::min(
                // deletion
                tmp + 1,
                core::cmp::min(
                    // insertion
                    cur[j] + 1,
                    // match or substitution
                    pre + usize::from(c_x != c_y),
                ),
            );
            pre = tmp;
        }
    }
    cur[y.len()]
}

/// Computes the Hamming distance between two strings.
///
/// The Hamming distance is defined as the number of positions at which
/// the corresponding symbols are different. It is named after
/// Richard Hamming, who introduced it in his fundamental paper on
/// Hamming codes.
///
/// While the input strings are not required to be of the same length, the
/// distance will only be computed up to the length of the shorter string.
///
/// # Arguments
///
/// * `x`: The first string.
/// * `y`: The second string.
///
/// # Examples
///
/// ```
/// use distances::strings::hamming;
///
/// let x = "NAJIBEATSPEPPERS";
/// let y = "NAJIBPEPPERSEATS";
///
/// let distance: u16 = hamming(x, y);
///
/// assert_eq!(distance, 10);
///
/// let x = "TOMEATSWHATFOODEATS";
/// let y = "FOODEATSWHATTOMEATS";
///
/// let distance: u16 = hamming(x, y);
///
/// assert_eq!(distance, 13);
/// ```
///
/// # References
///
/// * [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance)
/// * [Hamming's paper](https://doi.org/10.1002/j.1538-7305.1950.tb00463.x)
#[must_use]
pub fn hamming<U: UInt>(x: &str, y: &str) -> U {
    U::from(x.chars().zip(y.chars()).filter(|(a, b)| a != b).count())
}
