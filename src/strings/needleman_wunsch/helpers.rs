//! Helper functions for the Needleman-Wunsch algorithm.

use super::Penalties;
use crate::number::UInt;

/// The direction of best alignment at a given position in the DP table
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    /// Diagonal (Up and Left) for a match.
    D,
    /// Up for a gap in the first sequence.
    U,
    /// Left for a gap in the second sequence.
    L,
}

/// The type of edit needed to turn one sequence into another.
pub enum Edit {
    /// Delete a character at the given index.
    Del(usize),
    /// Insert a character at the given index.
    Ins(usize, char),
    /// Substitute a character at the given index.
    Sub(usize, char),
}

/// Computes the Needleman-Wunsch dynamic programming table for two sequences.
///
/// The penalty for a mismatch is 1, and the penalty for a gap is 1. There is no
/// penalty for a match. Our implementation minimizes the total penalty.
///
/// # Arguments
///
/// * `x`: The first sequence.
/// * `y`: The second sequence.
///
/// # Returns
///
/// A nested vector of tuples of total-penalty and Direction, representing the
/// best alignment at each position.
pub fn compute_table<U: UInt>(x: &str, y: &str, penalties: Penalties<U>) -> Vec<Vec<(U, Dir)>> {
    // Initializing table; the inner vectors represent rows in the table.
    let mut table = vec![vec![(U::zero(), Dir::D); x.len() + 1]; y.len() + 1];

    // The top-left cell starts with a total penalty of zero and no direction.
    table[0][0] = (U::zero(), Dir::D);

    // Initialize left-most column of distance values.
    for (i, row) in table.iter_mut().enumerate().skip(1) {
        row[0] = (penalties.gap * U::from(i), Dir::U);
    }

    // Initialize top row of distance values.
    for (j, cell) in table[0].iter_mut().enumerate().skip(1) {
        *cell = (penalties.gap * U::from(j), Dir::L);
    }

    // Set values for the body of the table
    for (i, y_c) in y.chars().enumerate() {
        for (j, x_c) in x.chars().enumerate() {
            // Check if sequences match at position `i` in `x` and `j` in `y`.
            let mismatch_penalty = if x_c == y_c {
                penalties.match_
            } else {
                penalties.mismatch
            };

            // Compute the three possible penalties and use the minimum to set
            // the value for the next entry in the table.
            let d00 = (table[i][j].0 + mismatch_penalty, Dir::D);
            let d01 = (table[i][j + 1].0 + penalties.gap, Dir::U);
            let d10 = (table[i + 1][j].0 + penalties.gap, Dir::L);

            table[i + 1][j + 1] = min2(d00, min2(d01, d10));
        }
    }

    table
}

/// Returns the minimum of two penalties, defaulting to the first input.
fn min2<U: UInt>(a: (U, Dir), b: (U, Dir)) -> (U, Dir) {
    if a.0 <= b.0 {
        a
    } else {
        b
    }
}

/// Converts two aligned sequences into vectors of edits.
///
/// # Arguments
///
/// * `x`: The first aligned sequence.
/// * `y`: The second aligned sequence.
///
/// # Returns
///
/// A 2-slice of Vec<Edit>, each containing the edits needed to convert one
/// sequence into the other
pub fn compute_edits(x: &str, y: &str) -> [Vec<Edit>; 2] {
    [_x_to_y(x, y), _x_to_y(y, x)]
}

/// Helper for `compute_edits` to compute the edits for `x` into `y`.
fn _x_to_y(x: &str, y: &str) -> Vec<Edit> {
    x.chars()
        .zip(y.chars())
        .filter(|(x, y)| x != y)
        .enumerate()
        .map(|(i, (x, y))| {
            if y == '-' {
                Edit::Del(i)
            } else if x == '-' {
                Edit::Ins(i, y)
            } else {
                Edit::Sub(i, y)
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
/// * `table`: The Needleman-Wunsch table.
/// * `[x, y]`: The two sequences to align.
///
/// # Returns
///
/// A tuple of the two aligned sequences.
pub fn trace_back_iterative<U: UInt>(
    table: &[Vec<(U, Dir)>],
    [x, y]: [&str; 2],
) -> (String, String) {
    let (x, y) = (x.as_bytes(), y.as_bytes());

    let (mut row_i, mut col_i) = (y.len(), x.len());
    let (mut aligned_x, mut aligned_y) = (Vec::new(), Vec::new());

    while row_i > 0 && col_i > 0 {
        match table[row_i][col_i].1 {
            Dir::D => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(y[row_i - 1]);
                row_i -= 1;
                col_i -= 1;
            }
            Dir::L => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(b'-');
                col_i -= 1;
            }
            Dir::U => {
                aligned_x.push(b'-');
                aligned_y.push(y[row_i - 1]);
                row_i -= 1;
            }
        }
    }

    aligned_x.reverse();
    aligned_y.reverse();

    let aligned_x = String::from_utf8(aligned_x)
        .unwrap_or_else(|_| unreachable!("We know we added valid characters."));
    let aligned_y = String::from_utf8(aligned_y)
        .unwrap_or_else(|_| unreachable!("We know we added valid characters."));

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
/// * `table`: The Needleman-Wunsch table.
/// * `[x, y]`: The two sequences to align.
///
/// # Returns
///
/// A tuple of the two aligned sequences.
pub fn trace_back_recursive<U: UInt>(
    table: &[Vec<(U, Dir)>],
    [x, y]: [&str; 2],
) -> (String, String) {
    let (mut aligned_x, mut aligned_y) = (Vec::new(), Vec::new());

    _trace_back_recursive(
        table,
        [y.len(), x.len()],
        [x.as_bytes(), y.as_bytes()],
        [&mut aligned_x, &mut aligned_y],
    );

    aligned_x.reverse();
    aligned_y.reverse();

    let aligned_x = String::from_utf8(aligned_x)
        .unwrap_or_else(|_| unreachable!("We know we added valid characters."));
    let aligned_y = String::from_utf8(aligned_y)
        .unwrap_or_else(|_| unreachable!("We know we added valid characters."));

    (aligned_x, aligned_y)
}

/// Helper function for `trace_back_recursive`.
///
/// # Arguments
///
/// * `table`: The Needleman-Wunsch table.
/// * `[row_i, col_i]`: mutable indices into the table.
/// * `[x, y]`: The two sequences to align, passed as slices of bytes.
/// * `[aligned_x, aligned_y]`: mutable aligned sequences that will be built
/// up from initially empty vectors.
fn _trace_back_recursive<U: UInt>(
    table: &[Vec<(U, Dir)>],
    [mut row_i, mut col_i]: [usize; 2],
    [x, y]: [&[u8]; 2],
    [aligned_x, aligned_y]: [&mut Vec<u8>; 2],
) {
    if row_i > 0 || col_i > 0 {
        match table[row_i][col_i].1 {
            Dir::D => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(y[row_i - 1]);
                row_i -= 1;
                col_i -= 1;
            }
            Dir::L => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(b'-');
                col_i -= 1;
            }
            Dir::U => {
                aligned_x.push(b'-');
                aligned_y.push(y[row_i - 1]);
                row_i -= 1;
            }
        };
        _trace_back_recursive(table, [row_i, col_i], [x, y], [aligned_x, aligned_y]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_table() {
        let x = "NAJIBPEPPERSEATS";
        let y = "NAJIBEATSPEPPERS";
        let table = compute_table::<u16>(x, y, Penalties::default());

        #[rustfmt::skip]
        let true_table: [[(u16, Dir); 17]; 17] = [
            [( 0, Dir::D), ( 1, Dir::L), ( 2, Dir::L), ( 3, Dir::L), ( 4, Dir::L), ( 5, Dir::L), ( 6, Dir::L), (7, Dir::L), (8, Dir::L), (9, Dir::L), (10, Dir::L), (11, Dir::L), (12, Dir::L), (13, Dir::L), (14, Dir::L), (15, Dir::L), (16, Dir::L)],
            [( 1, Dir::U), ( 0, Dir::D), ( 1, Dir::L), ( 2, Dir::L), ( 3, Dir::L), ( 4, Dir::L), ( 5, Dir::L), (6, Dir::L), (7, Dir::L), (8, Dir::L), ( 9, Dir::L), (10, Dir::L), (11, Dir::L), (12, Dir::L), (13, Dir::L), (14, Dir::L), (15, Dir::L)],
            [( 2, Dir::U), ( 1, Dir::U), ( 0, Dir::D), ( 1, Dir::L), ( 2, Dir::L), ( 3, Dir::L), ( 4, Dir::L), (5, Dir::L), (6, Dir::L), (7, Dir::L), ( 8, Dir::L), ( 9, Dir::L), (10, Dir::L), (11, Dir::L), (12, Dir::D), (13, Dir::L), (14, Dir::L)],
            [( 3, Dir::U), ( 2, Dir::U), ( 1, Dir::U), ( 0, Dir::D), ( 1, Dir::L), ( 2, Dir::L), ( 3, Dir::L), (4, Dir::L), (5, Dir::L), (6, Dir::L), ( 7, Dir::L), ( 8, Dir::L), ( 9, Dir::L), (10, Dir::L), (11, Dir::L), (12, Dir::L), (13, Dir::L)],
            [( 4, Dir::U), ( 3, Dir::U), ( 2, Dir::U), ( 1, Dir::U), ( 0, Dir::D), ( 1, Dir::L), ( 2, Dir::L), (3, Dir::L), (4, Dir::L), (5, Dir::L), ( 6, Dir::L), ( 7, Dir::L), ( 8, Dir::L), ( 9, Dir::L), (10, Dir::L), (11, Dir::L), (12, Dir::L)],
            [( 5, Dir::U), ( 4, Dir::U), ( 3, Dir::U), ( 2, Dir::U), ( 1, Dir::U), ( 0, Dir::D), ( 1, Dir::L), (2, Dir::L), (3, Dir::L), (4, Dir::L), ( 5, Dir::L), ( 6, Dir::L), ( 7, Dir::L), ( 8, Dir::L), ( 9, Dir::L), (10, Dir::L), (11, Dir::L)],
            [( 6, Dir::U), ( 5, Dir::U), ( 4, Dir::U), ( 3, Dir::U), ( 2, Dir::U), ( 1, Dir::U), ( 1, Dir::D), (1, Dir::D), (2, Dir::L), (3, Dir::L), ( 4, Dir::D), ( 5, Dir::L), ( 6, Dir::L), ( 7, Dir::D), ( 8, Dir::L), ( 9, Dir::L), (10, Dir::L)],
            [( 7, Dir::U), ( 6, Dir::U), ( 5, Dir::D), ( 4, Dir::U), ( 3, Dir::U), ( 2, Dir::U), ( 2, Dir::D), (2, Dir::D), (2, Dir::D), (3, Dir::D), ( 4, Dir::D), ( 5, Dir::D), ( 6, Dir::D), ( 7, Dir::D), ( 7, Dir::D), ( 8, Dir::L), ( 9, Dir::L)],
            [( 8, Dir::U), ( 7, Dir::U), ( 6, Dir::U), ( 5, Dir::U), ( 4, Dir::U), ( 3, Dir::U), ( 3, Dir::D), (3, Dir::D), (3, Dir::D), (3, Dir::D), ( 4, Dir::D), ( 5, Dir::D), ( 6, Dir::D), ( 7, Dir::D), ( 8, Dir::D), ( 7, Dir::D), ( 8, Dir::L)],
            [( 9, Dir::U), ( 8, Dir::U), ( 7, Dir::U), ( 6, Dir::U), ( 5, Dir::U), ( 4, Dir::U), ( 4, Dir::D), (4, Dir::D), (4, Dir::D), (4, Dir::D), ( 4, Dir::D), ( 5, Dir::D), ( 5, Dir::D), ( 6, Dir::L), ( 7, Dir::L), ( 8, Dir::U), ( 7, Dir::D)],
            [(10, Dir::U), ( 9, Dir::U), ( 8, Dir::U), ( 7, Dir::U), ( 6, Dir::U), ( 5, Dir::U), ( 4, Dir::D), (5, Dir::D), (4, Dir::D), (4, Dir::D), ( 5, Dir::D), ( 5, Dir::D), ( 6, Dir::D), ( 6, Dir::D), ( 7, Dir::D), ( 8, Dir::D), ( 8, Dir::U)],
            [(11, Dir::U), (10, Dir::U), ( 9, Dir::U), ( 8, Dir::U), ( 7, Dir::U), ( 6, Dir::U), ( 5, Dir::U), (4, Dir::D), (5, Dir::U), (5, Dir::D), ( 4, Dir::D), ( 5, Dir::L), ( 6, Dir::D), ( 6, Dir::D), ( 7, Dir::D), ( 8, Dir::D), ( 9, Dir::D)],
            [(12, Dir::U), (11, Dir::U), (10, Dir::U), ( 9, Dir::U), ( 8, Dir::U), ( 7, Dir::U), ( 6, Dir::D), (5, Dir::U), (4, Dir::D), (5, Dir::D), ( 5, Dir::U), ( 5, Dir::D), ( 6, Dir::D), ( 7, Dir::D), ( 7, Dir::D), ( 8, Dir::D), ( 9, Dir::D)],
            [(13, Dir::U), (12, Dir::U), (11, Dir::U), (10, Dir::U), ( 9, Dir::U), ( 8, Dir::U), ( 7, Dir::D), (6, Dir::U), (5, Dir::D), (4, Dir::D), ( 5, Dir::L), ( 6, Dir::D), ( 6, Dir::D), ( 7, Dir::D), ( 8, Dir::D), ( 8, Dir::D), ( 9, Dir::D)],
            [(14, Dir::U), (13, Dir::U), (12, Dir::U), (11, Dir::U), (10, Dir::U), ( 9, Dir::U), ( 8, Dir::U), (7, Dir::D), (6, Dir::U), (5, Dir::U), ( 4, Dir::D), ( 5, Dir::L), ( 6, Dir::L), ( 6, Dir::D), ( 7, Dir::L), ( 8, Dir::L), ( 9, Dir::D)],
            [(15, Dir::U), (14, Dir::U), (13, Dir::U), (12, Dir::U), (11, Dir::U), (10, Dir::U), ( 9, Dir::U), (8, Dir::U), (7, Dir::U), (6, Dir::U), ( 5, Dir::U), ( 4, Dir::D), ( 5, Dir::L), ( 6, Dir::L), ( 7, Dir::D), ( 8, Dir::D), ( 9, Dir::D)],
            [(16, Dir::U), (15, Dir::U), (14, Dir::U), (13, Dir::U), (12, Dir::U), (11, Dir::U), (10, Dir::U), (9, Dir::U), (8, Dir::U), (7, Dir::U), ( 6, Dir::U), ( 5, Dir::U), ( 4, Dir::D), ( 5, Dir::L), ( 6, Dir::L), ( 7, Dir::L), ( 8, Dir::D)]
        ];

        assert_eq!(table, true_table);
    }

    #[test]
    fn test_trace_back() {
        let peppers_x = "NAJIBPEPPERSEATS";
        let peppers_y = "NAJIBEATSPEPPERS";
        let peppers_table = compute_table::<u16>(peppers_x, peppers_y, Penalties::default());

        let (aligned_x, aligned_y) = trace_back_recursive(&peppers_table, [peppers_x, peppers_y]);
        assert_eq!(aligned_x, "NAJIB-PEPPERSEATS");
        assert_eq!(aligned_y, "NAJIBEATSPEPPE-RS");

        let (aligned_x, aligned_y) = trace_back_iterative(&peppers_table, [peppers_x, peppers_y]);
        assert_eq!(aligned_x, "NAJIB-PEPPERSEATS");
        assert_eq!(aligned_y, "NAJIBEATSPEPPE-RS");

        let guilty_x = "NOTGUILTY";
        let guilty_y = "NOTGUILTY";
        let guilty_table = compute_table::<u16>(guilty_x, guilty_y, Penalties::default());

        let (aligned_x, aligned_y) = trace_back_recursive(&guilty_table, [guilty_x, guilty_y]);
        assert_eq!(aligned_x, "NOTGUILTY");
        assert_eq!(aligned_y, "NOTGUILTY");

        let (aligned_x, aligned_y) = trace_back_iterative(&guilty_table, [guilty_x, guilty_y]);
        assert_eq!(aligned_x, "NOTGUILTY");
        assert_eq!(aligned_y, "NOTGUILTY");
    }
}
