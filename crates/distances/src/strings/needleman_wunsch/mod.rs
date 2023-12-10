//! Needleman-Wunsch edit-distance between two strings.
//!
//! This implementation should not be considered stable.

mod helpers;

use super::Penalties;
use crate::number::UInt;

use helpers::{compute_edits, compute_table, trace_back_iterative, trace_back_recursive, Edit};

/// Use a custom set of penalties to create a function to that calculates the
/// Needleman-Wunsch edit distance between two strings using the specified
/// penalties.
///
/// * [Demo](https://bioboot.github.io/bimm143_W20/class-material/nw/)
/// * [Wikipedia](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm)
///
/// # Arguments:
///
/// * `penalties`: The penalties to use in the generated function.
///
/// # Returns:
///
/// A function with the same signature as `nw_distance`.
pub fn nw_distance_custom<U: UInt>(penalties: Penalties<U>) -> impl Fn(&str, &str) -> U {
    move |x: &str, y: &str| compute_table(x, y, penalties)[y.len()][x.len()].0
}

/// Calculate the edit distance between two strings using Needleman-Wunsch table.
/// This function is only accurate with a scoring scheme for which all penalties
/// are non-negative.
///
/// * [Demo](https://bioboot.github.io/bimm143_W20/class-material/nw/)
/// * [Wikipedia](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm)
///
/// # Arguments:
///
/// * `x`: unaligned sequence represented as a `String`
/// * `y`: unaligned sequence represented as a `String`
#[must_use]
pub fn nw_distance<U: UInt>(x: &str, y: &str) -> U {
    compute_table(x, y, Penalties::default())[y.len()][x.len()].0
}

/// Use a custom set of penalties to create a function to that calculates the
/// set of edits needed to turn one unaligned sequence into another, as well as
/// the NW edit distance between the two sequences.
///
/// # Arguments:
///
/// * `penalties`: The penalties to use in the generated function.
///
/// # Returns:
///
/// A function with the same signature as `edits_recursive`.
pub fn edits_recursive_custom<U: UInt>(
    penalties: Penalties<U>,
) -> impl Fn(&str, &str) -> ([Vec<Edit>; 2], U) {
    move |x: &str, y: &str| {
        let table = compute_table(x, y, penalties);
        let (aligned_x, aligned_y) = trace_back_recursive(&table, [x, y]);
        (
            compute_edits(&aligned_x, &aligned_y),
            table[y.len()][x.len()].0,
        )
    }
}

/// Determine the set of edits needed to turn one unaligned sequence into
/// another, as well as the edit distance between the two sequences.
///
/// Contrast to `edits_iterative`, which uses the iterative trace back function.
///
/// For now, in cases where there exist ties for the shortest edit distance, we
/// only return one alignment.
///
/// # Arguments:
///
/// * `x`: an unaligned sequence.
/// * `y`: an unaligned sequence.
#[must_use]
pub fn edits_recursive<U: UInt>(x: &str, y: &str) -> ([Vec<Edit>; 2], U) {
    let table = compute_table(x, y, Penalties::default());
    let (aligned_x, aligned_y) = trace_back_recursive(&table, [x, y]);
    (
        compute_edits(&aligned_x, &aligned_y),
        table[y.len()][x.len()].0,
    )
}

/// Use a custom set of penalties to create a function to that calculates the
/// set of edits needed to turn one unaligned sequence into another, as well as
/// the NW edit distance between the two sequences.
///
/// # Arguments:
///
/// * `penalties`: The penalties to use in the generated function.
///
/// # Returns:
///
/// A function with the same signature as `edits_iterative`.
pub fn edits_iterative_custom<U: UInt>(
    penalties: Penalties<U>,
) -> impl Fn(&str, &str) -> ([Vec<Edit>; 2], U) {
    move |x: &str, y: &str| {
        let table = compute_table(x, y, penalties);
        let (aligned_x, aligned_y) = trace_back_iterative(&table, [x, y]);
        (
            compute_edits(&aligned_x, &aligned_y),
            table[y.len()][x.len()].0,
        )
    }
}

/// Determine the set of edits needed to turn one unaligned sequence into
/// another, as well as the edit distance between the two sequences.
///
/// Contrast to `edits_recursive`, which uses the recursive, trace back function
///
/// For now, in cases where there exist ties for the shortest edit distance, we
/// only return one alignment.
///
/// # Arguments:
///
/// * `x`: an unaligned sequence.
/// * `y`: an unaligned sequence.
#[must_use]
pub fn edits_iterative<U: UInt>(x: &str, y: &str) -> ([Vec<Edit>; 2], U) {
    let table = compute_table(x, y, Penalties::default());
    let (aligned_x, aligned_y) = trace_back_iterative(&table, [x, y]);
    (
        compute_edits(&aligned_x, &aligned_y),
        table[y.len()][x.len()].0,
    )
}

#[cfg(test)]
mod tests {
    use super::nw_distance;

    #[test]
    fn distance() {
        let x = "NAJIBEATSPEPPERS".to_string();
        let y = "NAJIBPEPPERSEATS".to_string();
        let d: u8 = nw_distance(&x, &y);
        assert_eq!(d, 8);

        let x = "NOTGUILTY".to_string();
        let y = "NOTGUILTY".to_string();
        let d: u8 = nw_distance(&x, &y);
        assert_eq!(d, 0);
    }
}
