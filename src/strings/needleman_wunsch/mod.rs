//! This will soon be moved into the `distances` crate.

pub(crate) mod alignment_helpers;

use crate::Number;

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
pub fn nw_distance<U: Number>(x: &str, y: &str) -> U {
    let table = alignment_helpers::compute_table(x, y);
    let edit_distance: usize = table[table.len() - 1][table[0].len() - 1].0;

    U::from(edit_distance)
}

/// Determine the set of edits needed to turn one unaligned sequence into another,
/// as well as the edit distance between the two sequences.
///
/// Contrast to `with_edits_iterative`, which uses an iterative, as
/// opposed to recursive, traceback function
///
/// For now, in cases where there exist ties for the shortest edit distance, we only
/// return one alignment.
///
/// # Arguments:
///
/// * `x`: unaligned sequence represented as a `String`
/// * `y`: unaligned sequence represented as a `String`
#[must_use]
pub fn with_edits_recursive<U: Number>(x: &str, y: &str) -> ([Vec<alignment_helpers::Edit>; 2], U) {
    let table = alignment_helpers::compute_table(x, y);
    let (aligned_x, aligned_y) = alignment_helpers::trace_back_recursive(&table, (x, y));

    let edit_x_into_y = alignment_helpers::alignment_to_edits(&aligned_x, &aligned_y);
    let edit_y_into_x = alignment_helpers::alignment_to_edits(&aligned_y, &aligned_x);

    let edit_distance: usize = table[table.len() - 1][table[0].len() - 1].0;

    ([edit_x_into_y, edit_y_into_x], U::from(edit_distance))
}

/// Determine the set of edits needed to turn one unaligned sequence into another,
/// as well as the edit distance between the two sequences.
///
/// Contrast to `with_edits_recursive`, which uses a recursive, as
/// opposed to iterative, traceback function
///
/// For now, in cases where there exist ties for the shortest edit distance, we only
/// return one alignment.
///
/// # Arguments:
///
/// * `x`: unaligned sequence represented as a `String`
/// * `y`: unaligned sequence represented as a `String`
#[must_use]
pub fn with_edits_iterative<U: Number>(x: &str, y: &str) -> ([Vec<alignment_helpers::Edit>; 2], U) {
    let table = alignment_helpers::compute_table(x, y);
    let (aligned_x, aligned_y) = alignment_helpers::trace_back_iterative(&table, (x, y));

    let edit_x_into_y = alignment_helpers::alignment_to_edits(&aligned_x, &aligned_y);
    let edit_y_into_x = alignment_helpers::alignment_to_edits(&aligned_y, &aligned_x);

    let edit_distance: usize = table[table.len() - 1][table[0].len() - 1].0;

    ([edit_x_into_y, edit_y_into_x], U::from(edit_distance))
}

#[cfg(test)]
mod tests {
    use super::nw_distance;

    #[test]
    fn test_needleman_wunsch() {
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
