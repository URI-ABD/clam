//! This will soon be moved into the `distances` crate.

mod helpers;

use crate::number::UInt;

use helpers::{compute_edits, compute_table, trace_back_iterative, trace_back_recursive, Edit};

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
    let table = compute_table(x, y);
    table[table.len() - 1][table[0].len() - 1].0
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
    let table = compute_table(x, y);
    let (aligned_x, aligned_y) = trace_back_recursive(&table, [x, y]);
    (
        compute_edits(&aligned_x, &aligned_y),
        table[table.len() - 1][table[0].len() - 1].0,
    )
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
    let table = compute_table(x, y);
    let (aligned_x, aligned_y) = trace_back_iterative(&table, [x, y]);
    (
        compute_edits(&aligned_x, &aligned_y),
        table[table.len() - 1][table[0].len() - 1].0,
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
