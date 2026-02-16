//! Distance functions for running tests

use std::iter::Sum;

use abd_clam::DistanceValue;
use num_traits::Float;

pub fn absolute_difference<T: DistanceValue>(a: &T, b: &T) -> T {
    if a < b { *b - *a } else { *a - *b }
}

pub fn manhattan<I: AsRef<[T]>, T: DistanceValue>(a: &I, b: &I) -> T {
    a.as_ref().iter().zip(b.as_ref().iter()).map(|(x, y)| absolute_difference(x, y)).sum()
}

pub fn hypotenuse<T: DistanceValue, U: Float>(a: &(T, T), b: &(T, T)) -> U {
    let (a1, a2) = a;
    let (b1, b2) = b;
    let height = U::from(absolute_difference(a1, b1)).unwrap_or_else(|| unreachable!("Height must be a finite number"));
    let base = U::from(absolute_difference(a2, b2)).unwrap_or_else(|| unreachable!("Base must be a finite number"));
    (height * height + base * base).sqrt()
}

pub fn euclidean<I: AsRef<[T]>, T: DistanceValue, U: Float + Sum>(a: &I, b: &I) -> U {
    a.as_ref()
        .iter()
        .zip(b.as_ref().iter())
        .map(|(x, y)| absolute_difference(x, y))
        .map(|d| U::from(d).unwrap_or_else(|| unreachable!("Distance must be a finite number")))
        .map(|d| d * d)
        .sum::<U>()
        .sqrt()
}

/// Compute the Levenshtein edit distance between two strings.
pub fn lev_unaligned<S>(x: S, y: S) -> usize
where
    S: AsRef<str>,
{
    lev_helper(x.as_ref().chars(), y.as_ref().chars())
}

/// Compute the Levenshtein edit distance between two aligned sequences.
#[cfg(feature = "musals")]
pub fn lev_aligned(x: &abd_clam::musals::AlignedSequence, y: &abd_clam::musals::AlignedSequence) -> usize {
    lev_helper(x.iter(), y.iter())
}

/// Compute the Levenshtein edit distance between two sequences.
fn lev_helper<S1: Iterator<Item = char>, S2: Iterator<Item = char>>(x: S1, y: S2) -> usize {
    let y = y.collect::<Vec<_>>();

    // calculate edit distance
    let mut cur = (0..=y.len()).collect::<Vec<_>>();
    for (i, char_x) in x.enumerate().map(|(i, c)| (i + 1, c)) {
        // get first column for this row
        let mut pre = cur[0];
        cur[0] = i;
        for (j, &char_y) in y.iter().enumerate() {
            let tmp = cur[j + 1];
            cur[j + 1] = core::cmp::min(
                tmp + 1, // deletion
                core::cmp::min(
                    cur[j] + 1,                          // insertion
                    pre + usize::from(char_x != char_y), // match or substitution
                ),
            );
            pre = tmp;
        }
    }

    cur[y.len()]
}
