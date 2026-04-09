//! A number of distance functions commonly used with CLAM.

use super::{DistanceValue, FloatDistanceValue};

/// The euclidean distance.
pub fn euclidean<T, I1, I2>(a: &I1, b: &I2) -> T
where
    T: FloatDistanceValue,
    I1: AsRef<[T]>,
    I2: AsRef<[T]>,
{
    a.as_ref().iter().zip(b.as_ref().iter()).map(|(&x, &y)| (x - y) * (x - y)).sum::<T>().sqrt()
}

/// The squared euclidean distance.
pub fn squared_euclidean<T, I1, I2>(a: &I1, b: &I2) -> T
where
    T: DistanceValue,
    I1: AsRef<[T]>,
    I2: AsRef<[T]>,
{
    iter_absolute_differences(a, b).map(|d| d * d).sum()
}

/// The manhattan distance.
pub fn manhattan<T, I1, I2>(a: &I1, b: &I2) -> T
where
    T: DistanceValue,
    I1: AsRef<[T]>,
    I2: AsRef<[T]>,
{
    iter_absolute_differences(a, b).sum()
}

/// An iterator over the absolute differences of corresponding elements.
fn iter_absolute_differences<'a, T, I1, I2>(a: &'a I1, b: &'a I2) -> impl Iterator<Item = T> + 'a
where
    T: DistanceValue + 'a,
    I1: AsRef<[T]>,
    I2: AsRef<[T]>,
{
    a.as_ref().iter().zip(b.as_ref().iter()).map(|(&x, &y)| if x < y { y - x } else { x - y })
}

/// The cosine distance.
pub fn cosine<T, I1, I2>(a: &I1, b: &I2) -> T
where
    T: FloatDistanceValue,
    I1: AsRef<[T]>,
    I2: AsRef<[T]>,
{
    let magnitude_a = squared_magnitude(a).sqrt();
    let magnitude_b = squared_magnitude(b).sqrt();
    if magnitude_a < T::epsilon() || magnitude_b < T::epsilon() {
        // If either vector is zero, define cosine distance as 1
        T::one()
    } else {
        T::one() - dot_product(a, b) / (magnitude_a * magnitude_b)
    }
}

/// The dot-product.
fn dot_product<T, I1, I2>(a: &I1, b: &I2) -> T
where
    T: DistanceValue,
    I1: AsRef<[T]>,
    I2: AsRef<[T]>,
{
    a.as_ref().iter().zip(b.as_ref().iter()).map(|(&x, &y)| x * y).sum()
}

/// The squared magnitude.
fn squared_magnitude<T, I>(a: &I) -> T
where
    T: DistanceValue,
    I: AsRef<[T]>,
{
    a.as_ref().iter().map(|&x| x * x).sum()
}

/// The levenshtein edit distance between two strings.
#[cfg(feature = "musals")]
pub fn levenshtein_strings<S: AsRef<str>>(a: S, b: S) -> usize {
    levenshtein_chars(a.as_ref().chars(), b.as_ref().chars())
}

/// The levenshtein edit distance between two aligned sequences.
#[cfg(feature = "musals")]
#[must_use]
pub fn levenshtein_aligned(a: &crate::musals::AlignedSequence, b: &crate::musals::AlignedSequence) -> usize {
    levenshtein_chars(a.iter(), b.iter())
}

/// The levenshtein edit distance between two iterators over `char`s.
#[cfg(feature = "musals")]
pub fn levenshtein_chars<S1, S2>(a: S1, b: S2) -> usize
where
    S1: Iterator<Item = char>,
    S2: Iterator<Item = char>,
{
    let b = b.collect::<Vec<_>>();

    // calculate edit distance
    let mut cur = (0..=b.len()).collect::<Vec<_>>();
    for (i, char_x) in a.enumerate().map(|(i, c)| (i + 1, c)) {
        // get first column for this row
        let mut pre = cur[0];
        cur[0] = i;
        for (j, &char_y) in b.iter().enumerate() {
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

    cur[b.len()]
}
