//! String distance metrics.

// use alloc::vec::Vec;  // no-std

use crate::number::UInt;

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
/// * `a` - The first string.
/// * `b` - The second string.
///
/// # Examples
///
/// ```
/// use distances::strings::levenshtein;
///
/// let a = "NAJIBEATSPEPPERS";
/// let b = "NAJIBPEPPERSEATS";
///
/// let distance: u16 = levenshtein(a, b);
///
/// assert_eq!(distance, 8);
///
/// let a = "TOMEATSWHATFOODEATS";
/// let b = "FOODEATSWHATTOMEATS";
///
/// let distance: u16 = levenshtein(a, b);
///
/// assert_eq!(distance, 6);
/// ```
///
/// # References
///
/// * [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
#[must_use]
pub fn levenshtein<U: UInt>(a: &str, b: &str) -> U {
    U::from(if a.is_empty() {
        // handle special case of 0 length
        b.len()
    } else if b.is_empty() {
        // handle special case of 0 length
        a.len()
    } else if a.len() < b.len() {
        // require that a is no shorter than b
        _levenshtein(b, a)
    } else {
        _levenshtein(a, b)
    })
}

/// Helper for Levenshtein distance.
fn _levenshtein(a: &str, b: &str) -> usize {
    // initialize DP table for string b
    let mut cur: Vec<usize> = (0..=b.len()).collect();

    // calculate edit distance
    for (i, ca) in a.chars().enumerate() {
        // get first column for this row
        let mut pre = cur[0];
        cur[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let tmp = cur[j + 1];
            cur[j + 1] = core::cmp::min(
                // deletion
                tmp + 1,
                core::cmp::min(
                    // insertion
                    cur[j] + 1,
                    // match or substitution
                    pre + usize::from(ca != cb),
                ),
            );
            pre = tmp;
        }
    }
    cur[b.len()]
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
/// * `x` - The first string.
/// * `y` - The second string.
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
