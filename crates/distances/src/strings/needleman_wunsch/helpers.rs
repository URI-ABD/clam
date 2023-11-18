//! Helper functions for the Needleman-Wunsch algorithm.

use crate::{number::UInt, strings::Penalties};

/// The direction of best alignment at a given position in the DP table
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Diagonal (Up and Left) for a match.
    Diagonal,
    /// Up for a gap in the first sequence.
    Up,
    /// Left for a gap in the second sequence.
    Left,
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
pub fn compute_table<U: UInt>(
    x: &str,
    y: &str,
    penalties: Penalties<U>,
) -> Vec<Vec<(U, Direction)>> {
    // Initializing table; the inner vectors represent rows in the table.
    let mut table = vec![vec![(U::zero(), Direction::Diagonal); x.len() + 1]; y.len() + 1];

    // The top-left cell starts with a total penalty of zero and no direction.
    table[0][0] = (U::zero(), Direction::Diagonal);

    // Initialize left-most column of distance values.
    for (i, row) in table.iter_mut().enumerate().skip(1) {
        row[0] = (penalties.gap * U::from(i), Direction::Up);
    }

    // Initialize top row of distance values.
    for (j, cell) in table[0].iter_mut().enumerate().skip(1) {
        *cell = (penalties.gap * U::from(j), Direction::Left);
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
            let d00 = (table[i][j].0 + mismatch_penalty, Direction::Diagonal);
            let d01 = (table[i][j + 1].0 + penalties.gap, Direction::Up);
            let d10 = (table[i + 1][j].0 + penalties.gap, Direction::Left);

            table[i + 1][j + 1] = min2(d00, min2(d01, d10));
        }
    }

    table
}

/// Returns the minimum of two penalties, defaulting to the first input.
fn min2<U: UInt>(a: (U, Direction), b: (U, Direction)) -> (U, Direction) {
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
    // TODO: Revisit this for correctness when we start working on the MSA.
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
    table: &[Vec<(U, Direction)>],
    [x, y]: [&str; 2],
) -> (String, String) {
    let (x, y) = (x.as_bytes(), y.as_bytes());

    let (mut row_i, mut col_i) = (y.len(), x.len());
    let (mut aligned_x, mut aligned_y) = (Vec::new(), Vec::new());

    while row_i > 0 && col_i > 0 {
        match table[row_i][col_i].1 {
            Direction::Diagonal => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(y[row_i - 1]);
                row_i -= 1;
                col_i -= 1;
            }
            Direction::Left => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(b'-');
                col_i -= 1;
            }
            Direction::Up => {
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
    table: &[Vec<(U, Direction)>],
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
    table: &[Vec<(U, Direction)>],
    [mut row_i, mut col_i]: [usize; 2],
    [x, y]: [&[u8]; 2],
    [aligned_x, aligned_y]: [&mut Vec<u8>; 2],
) {
    if row_i > 0 || col_i > 0 {
        match table[row_i][col_i].1 {
            Direction::Diagonal => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(y[row_i - 1]);
                row_i -= 1;
                col_i -= 1;
            }
            Direction::Left => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(b'-');
                col_i -= 1;
            }
            Direction::Up => {
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
        let true_table: [[(u16, Direction); 17]; 17] = [
            [( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), ( 4, Direction::Left    ), ( 5, Direction::Left    ), ( 6, Direction::Left    ), (7, Direction::Left    ), (8, Direction::Left    ), (9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    ), (13, Direction::Left    ), (14, Direction::Left    ), (15, Direction::Left    ), (16, Direction::Left    )],
            [( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), ( 4, Direction::Left    ), ( 5, Direction::Left    ), (6, Direction::Left    ), (7, Direction::Left    ), (8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    ), (13, Direction::Left    ), (14, Direction::Left    ), (15, Direction::Left    )],
            [( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), ( 4, Direction::Left    ), (5, Direction::Left    ), (6, Direction::Left    ), (7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Diagonal), (13, Direction::Left    ), (14, Direction::Left    )],
            [( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), (4, Direction::Left    ), (5, Direction::Left    ), (6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    ), (13, Direction::Left    )],
            [( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), (3, Direction::Left    ), (4, Direction::Left    ), (5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    )],
            [( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), (2, Direction::Left    ), (3, Direction::Left    ), (4, Direction::Left    ), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    )],
            [( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 1, Direction::Diagonal), (1, Direction::Diagonal), (2, Direction::Left    ), (3, Direction::Left    ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Diagonal), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    )],
            [( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Diagonal), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 2, Direction::Diagonal), (2, Direction::Diagonal), (2, Direction::Diagonal), (3, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Left    ), ( 9, Direction::Left    )],
            [( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 3, Direction::Diagonal), (3, Direction::Diagonal), (3, Direction::Diagonal), (3, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Left    )],
            [( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 4, Direction::Diagonal), (4, Direction::Diagonal), (4, Direction::Diagonal), (4, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Up      ), ( 7, Direction::Diagonal)],
            [(10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Diagonal), (5, Direction::Diagonal), (4, Direction::Diagonal), (4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 8, Direction::Up      )],
            [(11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), (4, Direction::Diagonal), (5, Direction::Up      ), (5, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Diagonal), (5, Direction::Up      ), (4, Direction::Diagonal), (5, Direction::Diagonal), ( 5, Direction::Up      ), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Diagonal), (6, Direction::Up      ), (5, Direction::Diagonal), (4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(14, Direction::Up      ), (13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), (7, Direction::Diagonal), (6, Direction::Up      ), (5, Direction::Up      ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 6, Direction::Diagonal), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Diagonal)],
            [(15, Direction::Up      ), (14, Direction::Up      ), (13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), (8, Direction::Up      ), (7, Direction::Up      ), (6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(16, Direction::Up      ), (15, Direction::Up      ), (14, Direction::Up      ), (13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), (9, Direction::Up      ), (8, Direction::Up      ), (7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Diagonal)]
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
