//! Helper functions for the Needleman-Wunsch algorithm.

/// Enum representing the direction of best alignment at a given position in the dp table
/// (used for trace back)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Diagonal (Up and Left) for a match.
    Diagonal,
    /// Up for a gap in the first sequence.
    Up,
    /// Left for a gap in the second sequence.
    Left,
    /// None for the start of the table.
    None,
}

/// Returns the minimum of two tuples of usize and Direction by comparing the usize values.
const fn min2(a: (usize, Direction), b: (usize, Direction)) -> (usize, Direction) {
    if a.0 < b.0 {
        a
    } else {
        b
    }
}

/// Computes the Needleman-Wunsch dynamic programming table for two sequences.
///
/// The penalty for a mismatch is 1, and the penalty for a gap is 1. There is no
/// penalty for a match.
///
/// # Arguments
///
/// * `x` - The first sequence.
/// * `y` - The second sequence.
///
/// # Returns
///
/// A 2D vector of tuples of usize and Direction, where the usize represents the
/// edit distance at that position in the table, and the Direction represents the
/// direction of the best alignment at that position.
pub fn compute_table(x: &str, y: &str) -> Vec<Vec<(usize, Direction)>> {
    let len_x = x.len();
    let len_y = y.len();

    // Initializing table; subvecs represent rows
    let mut table = vec![vec![(0, Direction::None); len_x + 1]; len_y + 1];

    let gap_penalty = 1;

    // Initialize top row and left column distance values
    for (i, row) in table.iter_mut().enumerate() {
        row[0] = (gap_penalty * i, Direction::Up);
    }

    for (j, cell) in table[0].iter_mut().enumerate() {
        *cell = (gap_penalty * j, Direction::Left);
    }

    table[0][0] = (0, Direction::None);

    // Set values for the body of the table
    for (i, y_c) in y.chars().enumerate() {
        for (j, x_c) in x.chars().enumerate() {
            // Check if sequences match at position i in x and j in y
            // Reason for subtraction is that NW considers an artificial gap at the start
            // of each sequence, so the dp tables' indices are 1 higher than that of
            // the actual sequences
            let mismatch_penalty = usize::from(x_c != y_c);

            let d00 = (table[i][j].0 + mismatch_penalty, Direction::Diagonal);
            let d01 = (table[i][j + 1].0 + gap_penalty, Direction::Up);
            let d10 = (table[i + 1][j].0 + gap_penalty, Direction::Left);

            table[i + 1][j + 1] = min2(min2(d10, d01), d00);
        }
    }

    table
}

/// Enum representing the type of edit needed to turn one sequence into another.
pub enum Edit {
    /// Deletion of a character at the given index.
    Deletion(usize),
    /// Insertion of a character at the given index.
    Insertion(usize, char),
    /// Substitution of a character at the given index.
    Substitution(usize, char),
}

/// Converts two aligned sequences into a vector of edits.
///
/// # Arguments
///
/// * `aligned_x` - The first aligned sequence.
/// * `aligned_y` - The second aligned sequence.
///
/// # Returns
///
/// A vector of `Edit`s needed to turn the first sequence into the second.
#[allow(clippy::ptr_arg)]
pub fn alignment_to_edits(aligned_x: &str, aligned_y: &str) -> Vec<Edit> {
    aligned_x
        .chars()
        .zip(aligned_y.chars())
        .filter(|(x, y)| x != y)
        .enumerate()
        .map(|(index, (x, y))| {
            if x == '-' {
                Edit::Insertion(index, y)
            } else if y == '-' {
                Edit::Deletion(index)
            } else {
                Edit::Substitution(index, y)
            }
        })
        .collect()
}

/// Iteratively traces back through the Needleman-Wunsch table to get the alignment of two sequences.
///
/// For now, we ignore ties in the paths that can be followed to get the best alignments.
/// We break ties by always choosing the `Diagonal` path first, then the `Left`
/// path, then the `Up` path.
///
/// # Arguments
///
/// * `table` - The Needleman-Wunsch table.
/// * `unaligned_seqs` - A tuple of the two sequences to align.
///
/// # Returns
///
/// A tuple of the two aligned sequences.
pub fn trace_back_iterative(
    table: &Vec<Vec<(usize, Direction)>>,
    unaligned_seqs: (&str, &str),
) -> (String, String) {
    let (unaligned_x, unaligned_y) = unaligned_seqs;
    let unaligned_x = unaligned_x.as_bytes();
    let unaligned_y = unaligned_y.as_bytes();

    let mut row_index = table.len() - 1;
    let mut column_index = table[0].len() - 1;

    let (mut aligned_x, mut aligned_y) = (Vec::new(), Vec::new());
    let mut direction = table[row_index][column_index].1;

    while direction != Direction::None {
        match direction {
            Direction::Diagonal => {
                aligned_x.push(unaligned_x[column_index - 1]);
                aligned_y.push(unaligned_y[row_index - 1]);
                row_index -= 1;
                column_index -= 1;
            }
            Direction::Left => {
                aligned_x.push(unaligned_x[column_index - 1]);
                aligned_y.push(b'-');
                column_index -= 1;
            }
            Direction::Up => {
                aligned_x.push(b'-');
                aligned_y.push(unaligned_y[row_index - 1]);
                row_index -= 1;
            }
            Direction::None => {}
        }

        direction = table[row_index][column_index].1;
    }

    aligned_x.reverse();
    aligned_y.reverse();

    let aligned_x = String::from_utf8(aligned_x).unwrap_or_else(|_| unreachable!("Invalid UTF-8"));
    let aligned_y = String::from_utf8(aligned_y).unwrap_or_else(|_| unreachable!("Invalid UTF-8"));

    (aligned_x, aligned_y)
}

/// Recursively traces back through the Needleman-Wunsch table to get the alignment of two sequences.
///
/// For now, we ignore ties in the paths that can be followed to get the best alignments.
/// We break ties by always choosing the `Diagonal` path first, then the `Left`
/// path, then the `Up` path.
///
/// # Arguments
///
/// * `table` - The Needleman-Wunsch table.
/// * `unaligned_seqs` - A tuple of the two sequences to align.
///
/// # Returns
///
/// A tuple of the two aligned sequences.
pub fn trace_back_recursive(
    table: &Vec<Vec<(usize, Direction)>>,
    unaligned_seqs: (&str, &str),
) -> (String, String) {
    let (unaligned_x, unaligned_y) = unaligned_seqs;
    let (mut aligned_x, mut aligned_y) = (Vec::new(), Vec::new());

    _trace_back_recursive(
        table,
        (table.len() - 1, table[0].len() - 1),
        (unaligned_x.as_bytes(), unaligned_y.as_bytes()),
        (&mut aligned_x, &mut aligned_y),
    );

    let aligned_x = String::from_utf8(aligned_x).unwrap_or_else(|_| unreachable!("Invalid UTF-8"));
    let aligned_y = String::from_utf8(aligned_y).unwrap_or_else(|_| unreachable!("Invalid UTF-8"));

    (aligned_x, aligned_y)
}

/// Helper function for `trace_back_recursive`.
fn _trace_back_recursive(
    table: &Vec<Vec<(usize, Direction)>>,
    (mut row_index, mut column_index): (usize, usize),
    (unaligned_x, unaligned_y): (&[u8], &[u8]),
    (aligned_x, aligned_y): (&mut Vec<u8>, &mut Vec<u8>),
) {
    let direction = table[row_index][column_index].1;

    match direction {
        Direction::Diagonal => {
            aligned_x.push(unaligned_x[column_index - 1]);
            aligned_y.push(unaligned_y[row_index - 1]);
            row_index -= 1;
            column_index -= 1;
        }
        Direction::Left => {
            aligned_x.push(unaligned_x[column_index - 1]);
            aligned_y.push(b'-');
            column_index -= 1;
        }
        Direction::Up => {
            aligned_x.push(b'-');
            aligned_y.push(unaligned_y[row_index - 1]);
            row_index -= 1;
        }
        Direction::None => {
            aligned_x.reverse();
            aligned_y.reverse();

            return;
        }
    };

    _trace_back_recursive(
        table,
        (row_index, column_index),
        (unaligned_x, unaligned_y),
        (aligned_x, aligned_y),
    );
}

#[cfg(test)]
mod tests {
    use super::compute_table;
    use super::trace_back_iterative;
    use super::trace_back_recursive;
    use super::Direction;

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_compute_table() {
        let x = "NAJIBPEPPERSEATS".to_string();
        let y = "NAJIBEATSPEPPERS".to_string();
        let table = compute_table(&x, &y);
        assert_eq!(
            table,
            [
                [
                    (0, Direction::None),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left),
                    (12, Direction::Left),
                    (13, Direction::Left),
                    (14, Direction::Left),
                    (15, Direction::Left),
                    (16, Direction::Left)
                ],
                [
                    (1, Direction::Up),
                    (0, Direction::Diagonal),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left),
                    (12, Direction::Left),
                    (13, Direction::Left),
                    (14, Direction::Left),
                    (15, Direction::Left)
                ],
                [
                    (2, Direction::Up),
                    (1, Direction::Up),
                    (0, Direction::Diagonal),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left),
                    (12, Direction::Diagonal),
                    (13, Direction::Left),
                    (14, Direction::Left)
                ],
                [
                    (3, Direction::Up),
                    (2, Direction::Up),
                    (1, Direction::Up),
                    (0, Direction::Diagonal),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left),
                    (12, Direction::Left),
                    (13, Direction::Left)
                ],
                [
                    (4, Direction::Up),
                    (3, Direction::Up),
                    (2, Direction::Up),
                    (1, Direction::Up),
                    (0, Direction::Diagonal),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left),
                    (12, Direction::Left)
                ],
                [
                    (5, Direction::Up),
                    (4, Direction::Up),
                    (3, Direction::Up),
                    (2, Direction::Up),
                    (1, Direction::Up),
                    (0, Direction::Diagonal),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left)
                ],
                [
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Up),
                    (3, Direction::Up),
                    (2, Direction::Up),
                    (1, Direction::Up),
                    (1, Direction::Diagonal),
                    (1, Direction::Diagonal),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Diagonal),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left)
                ],
                [
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Diagonal),
                    (4, Direction::Up),
                    (3, Direction::Up),
                    (2, Direction::Up),
                    (2, Direction::Diagonal),
                    (2, Direction::Diagonal),
                    (2, Direction::Diagonal),
                    (3, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Left),
                    (9, Direction::Left)
                ],
                [
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Up),
                    (3, Direction::Up),
                    (3, Direction::Diagonal),
                    (3, Direction::Diagonal),
                    (3, Direction::Diagonal),
                    (3, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Left)
                ],
                [
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Up),
                    (4, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Up),
                    (7, Direction::Diagonal)
                ],
                [
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (8, Direction::Up)
                ],
                [
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Up),
                    (5, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (9, Direction::Diagonal)
                ],
                [
                    (12, Direction::Up),
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Diagonal),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (5, Direction::Up),
                    (5, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (9, Direction::Diagonal)
                ],
                [
                    (13, Direction::Up),
                    (12, Direction::Up),
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Diagonal),
                    (6, Direction::Up),
                    (5, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (9, Direction::Diagonal)
                ],
                [
                    (14, Direction::Up),
                    (13, Direction::Up),
                    (12, Direction::Up),
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Diagonal),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (6, Direction::Diagonal),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Diagonal)
                ],
                [
                    (15, Direction::Up),
                    (14, Direction::Up),
                    (13, Direction::Up),
                    (12, Direction::Up),
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (9, Direction::Diagonal)
                ],
                [
                    (16, Direction::Up),
                    (15, Direction::Up),
                    (14, Direction::Up),
                    (13, Direction::Up),
                    (12, Direction::Up),
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Diagonal)
                ]
            ]
        );
    }

    #[test]
    fn test_traceback_recursive() {
        let peppers_x = "NAJIBPEPPERSEATS".to_string();
        let peppers_y = "NAJIBEATSPEPPERS".to_string();
        let peppers_table = compute_table(&peppers_x, &peppers_y);
        let (aligned_x, aligned_y) = trace_back_recursive(&peppers_table, (&peppers_x, &peppers_y));

        assert_eq!(aligned_x, "NAJIB-PEPPERSEATS");
        assert_eq!(aligned_y, "NAJIBEATSPEPPE-RS");

        let guilty_x = "NOTGUILTY".to_string();
        let guilty_y = "NOTGUILTY".to_string();
        let guilty_table = compute_table(&guilty_x, &guilty_y);
        let (aligned_x, aligned_y) = trace_back_recursive(&guilty_table, (&guilty_x, &guilty_y));

        assert_eq!(aligned_x, "NOTGUILTY");
        assert_eq!(aligned_y, "NOTGUILTY");
    }

    #[test]
    fn test_traceback_iterative() {
        let peppers_x = "NAJIBPEPPERSEATS".to_string();
        let peppers_y = "NAJIBEATSPEPPERS".to_string();
        let peppers_table = compute_table(&peppers_x, &peppers_y);
        let (aligned_x, aligned_y) = trace_back_iterative(&peppers_table, (&peppers_x, &peppers_y));

        assert_eq!(aligned_x, "NAJIB-PEPPERSEATS");
        assert_eq!(aligned_y, "NAJIBEATSPEPPE-RS");

        let guilty_x = "NOTGUILTY".to_string();
        let guilty_y = "NOTGUILTY".to_string();
        let guilty_table = compute_table(&guilty_x, &guilty_y);
        let (aligned_x, aligned_y) = trace_back_iterative(&guilty_table, (&guilty_x, &guilty_y));

        assert_eq!(aligned_x, "NOTGUILTY");
        assert_eq!(aligned_y, "NOTGUILTY");
    }
}
