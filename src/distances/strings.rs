use super::alignment_helpers;
use crate::number::Number;

pub fn levenshtein<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    let (len_x, len_y) = (x.len(), y.len());

    if len_x == 0 {
        // handle special case of 0 length
        U::from(len_y).unwrap()
    } else if len_y == 0 {
        // handle special case of 0 length
        U::from(len_x).unwrap()
    } else if len_x < len_y {
        // require len_a < len_b
        levenshtein(y, x)
    } else {
        let len_y = len_y + 1;

        // initialize DP table for string y
        let mut cur = (0..len_y).collect::<Vec<_>>();

        // calculate edit distance
        for (i, cx) in x.iter().enumerate() {
            // get first column for this row
            let mut pre = cur[0];
            cur[0] = i + 1;
            for (j, cy) in y.iter().enumerate() {
                let tmp = cur[j + 1];
                cur[j + 1] = std::cmp::min(
                    // deletion
                    tmp + 1,
                    std::cmp::min(
                        // insertion
                        cur[j] + 1,
                        // match or substitution
                        pre + if cx == cy { 0 } else { 1 },
                    ),
                );
                pre = tmp;
            }
        }

        U::from(cur[len_y - 1]).unwrap()
    }
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
/// * `x`: unaligned sequence represented as a slice of type T
/// * `y`: unaligned sequence represented as a slice of type T
pub fn needleman_wunsch<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    let table = alignment_helpers::compute_nw_table(x, y);
    let edit_distance: usize = table[table.len() - 1][table[0].len() - 1].0;

    U::from(edit_distance).unwrap()
}

/// Determine the set of edits needed to turn one unaligned sequence into another,
/// as well as the edit distance between the two sequences.
///
/// Contrast to `needleman_wunsch_with_edits_iterative`, which uses an iterative, as
/// opposed to recursive, traceback function
///
/// For now, in cases where there exist ties for the shortest edit distance, we only
/// return one alignment.
///
/// # Arguments:
///
/// * `x`: unaligned sequence represented as a slice of type T
/// * `y`: unaligned sequence represented as a slice of type T
pub fn needleman_wunsch_with_edits_recursive<T: Number, U: Number>(
    x: &[T],
    y: &[T],
) -> ([Vec<alignment_helpers::Edit<T>>; 2], U) {
    let table = alignment_helpers::compute_nw_table(x, y);
    let (aligned_x, aligned_y) = alignment_helpers::traceback_recursive(&table, (x, y));

    let edit_x_into_y = alignment_helpers::alignment_to_edits(&aligned_x, &aligned_y);
    let edit_y_into_x = alignment_helpers::alignment_to_edits(&aligned_y, &aligned_x);

    let edit_distance: usize = table[table.len() - 1][table[0].len() - 1].0;

    ([edit_x_into_y, edit_y_into_x], U::from(edit_distance).unwrap())
}

/// Determine the set of edits needed to turn one unaligned sequence into another,
/// as well as the edit distance between the two sequences.
///
/// Contrast to `needleman_wunsch_with_edits_recursive`, which uses a recursive, as
/// opposed to iterative, traceback function
///
/// For now, in cases where there exist ties for the shortest edit distance, we only
/// return one alignment.
///
/// # Arguments:
///
/// * `x`: unaligned sequence represented as a slice of type T
/// * `y`: unaligned sequence represented as a slice of type T
pub fn needleman_wunsch_with_edits_iterative<T: Number, U: Number>(
    x: &[T],
    y: &[T],
) -> ([Vec<alignment_helpers::Edit<T>>; 2], U) {
    let table = alignment_helpers::compute_nw_table(x, y);
    let (aligned_x, aligned_y) = alignment_helpers::traceback_iterative(&table, (x, y));

    let edit_x_into_y = alignment_helpers::alignment_to_edits(&aligned_x, &aligned_y);
    let edit_y_into_x = alignment_helpers::alignment_to_edits(&aligned_y, &aligned_x);

    let edit_distance: usize = table[table.len() - 1][table[0].len() - 1].0;

    ([edit_x_into_y, edit_y_into_x], U::from(edit_distance).unwrap())
}

#[cfg(test)]
mod tests {
    use super::levenshtein;
    use super::needleman_wunsch;

    #[test]
    fn test_levenshtein() {
        let x = [0, 1, 2, 2, 1, 3, 4];
        let y = [5, 1, 2, 2, 6, 3];

        let lev: i32 = levenshtein(&x, &y);

        assert_eq!(lev, 3)
    }

    #[test]
    fn test_needleman_wunsch() {
        let x_int = [0, 1, 2, 2, 1, 3, 4];
        let y_int = [5, 1, 2, 2, 6, 3];
        let nw_int: i32 = needleman_wunsch(&x_int, &y_int);
        assert_eq!(nw_int, 3);

        let x_u8 = "NAJIBEATSPEPPERS".as_bytes();
        let y_u8 = "NAJIBPEPPERSEATS".as_bytes();
        let nw_u8: u8 = needleman_wunsch(x_u8, y_u8);
        assert_eq!(nw_u8, 8);

        let x = "NOTGUILTY".as_bytes();
        let y = "NOTGUILTY".as_bytes();
        let nw: u8 = needleman_wunsch(x, y);
        assert_eq!(nw, 0);
    }
}
