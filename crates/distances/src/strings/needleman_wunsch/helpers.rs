//! Helper functions for the Needleman-Wunsch algorithm.

use serde::{Deserialize, Serialize};

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
#[derive(Clone, Debug, PartialEq, Eq, Copy, Serialize, Deserialize)]
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
/// Users can input custom penalities, but those penalties should satisfy the following:
/// * All penalties are non-negative
/// * The match "penalty" is 0
///
///  Our implementation *minimizes* the total penalty.
///
/// # Arguments
///
/// * `x`: The first sequence.
/// * `y`: The second sequence.
/// * `penalties`: The penalties for a match, mismatch, and gap.
///
/// # Returns
///
/// A nested vector of tuples of total-penalty and Direction, representing the
/// best alignment at each position.
pub fn compute_table<U: UInt>(x: &str, y: &str, penalties: Penalties<U>) -> Vec<Vec<(U, Direction)>> {
    // Initializing table; the inner vectors represent rows in the table.
    let mut table = vec![vec![(U::ZERO, Direction::Diagonal); x.len() + 1]; y.len() + 1];

    // The top-left cell starts with a total penalty of zero and no direction.
    table[0][0] = (U::ZERO, Direction::Diagonal);

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
            let mismatch_penalty = if x_c == y_c { penalties.match_ } else { penalties.mismatch };

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
    if a.0 <= b.0 { a } else { b }
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
#[must_use]
pub fn trace_back_iterative<U: UInt>(table: &[Vec<(U, Direction)>], [x, y]: [&str; 2]) -> (String, String) {
    let (x, y) = (x.as_bytes(), y.as_bytes());

    let (mut row_i, mut col_i) = (y.len(), x.len());
    let (mut aligned_x, mut aligned_y) = (Vec::new(), Vec::new());

    while row_i > 0 || col_i > 0 {
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

    let aligned_x = String::from_utf8(aligned_x).unwrap_or_else(|_| unreachable!("We know we added valid characters."));
    let aligned_y = String::from_utf8(aligned_y).unwrap_or_else(|_| unreachable!("We know we added valid characters."));

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
#[must_use]
pub fn trace_back_recursive<U: UInt>(table: &[Vec<(U, Direction)>], [x, y]: [&str; 2]) -> (String, String) {
    let (mut aligned_x, mut aligned_y) = (Vec::new(), Vec::new());

    trb_helper(table, [y.len(), x.len()], [x.as_bytes(), y.as_bytes()], [&mut aligned_x, &mut aligned_y]);

    aligned_x.reverse();
    aligned_y.reverse();

    let aligned_x = String::from_utf8(aligned_x).unwrap_or_else(|_| unreachable!("We know we added valid characters."));
    let aligned_y = String::from_utf8(aligned_y).unwrap_or_else(|_| unreachable!("We know we added valid characters."));

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
///   up from initially empty vectors.
fn trb_helper<U: UInt>(table: &[Vec<(U, Direction)>], [mut row_i, mut col_i]: [usize; 2], [x, y]: [&[u8]; 2], [aligned_x, aligned_y]: [&mut Vec<u8>; 2]) {
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
        }
        trb_helper(table, [row_i, col_i], [x, y], [aligned_x, aligned_y]);
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
/// A 2-slice of `Vec<Edit>`, each containing the edits needed to convert one aligned
/// sequence into the other.
/// Since both input sequences are aligned, all edits are substitutions in the returned vectors are Substitutions.
#[must_use]
pub fn compute_edits(x: &str, y: &str) -> [Vec<Edit>; 2] {
    [x2y_helper(x, y), x2y_helper(y, x)]
}

/// Helper for `compute_edits` to compute the edits for turning aligned `x` into aligned `y`.
///
/// Expects `x` and `y` to be aligned sequences generated by `trace_back_iterative` or `trace_back_recursive`.
/// Returns a vector of substitutions needed to turn `x` into `y`.
///
/// # Arguments
///
/// * `x`: The first aligned sequence.
/// * `y`: The second aligned sequence.
///
/// # Returns
///
/// A vector of edits needed to convert `x` into `y`.
#[must_use]
pub fn x2y_helper(x: &str, y: &str) -> Vec<Edit> {
    x.chars()
        .zip(y.chars())
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, (_, y))| Edit::Sub(i, y))
        .collect()
}

/// Given two aligned strings, returns into a sequence of edits to transform the unaligned
/// version of one into the unaligned version of the other.
///
/// Requires that gaps are never aligned with gaps (our NW implementation with the default penalties ensures this).
/// Expects to receive two strings which were aligned using either `trace_back_iterative` or `trace_back_recursive`.
///
/// # Arguments
///
/// * `x`: An aligned string.
/// * `y`: An aligned string.
///
/// # Returns
///
/// A vector of edits to transform the unaligned version of `x` into the unaligned version of `y`.
#[must_use]
pub fn unaligned_x_to_y(x: &str, y: &str) -> Vec<Edit> {
    let mut unaligned_x_to_y = Vec::new();
    let mut modifier = 0;

    x.chars()
        .zip(y.chars())
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .for_each(|(index, (c_x, c_y))| {
            let i = index - modifier;
            if c_x == '-' {
                unaligned_x_to_y.push(Edit::Ins(i, c_y));
            } else if c_y == '-' {
                unaligned_x_to_y.push(Edit::Del(i));
                modifier += 1;
            } else {
                unaligned_x_to_y.push(Edit::Sub(i, c_y));
            }
        });
    unaligned_x_to_y
}

/// Given two unaligned strings, returns into a sequence of edits to transform the unaligned
/// version of one into the unaligned version of the other.
///
/// Requires that gaps are never aligned with gaps (our NW implementation with the default penalties ensures this).
/// Expects to receive two strings which were aligned using either `trace_back_iterative` or `trace_back_recursive`.
///
/// # Arguments
///
/// * `x`: A unaligned string.
/// * `y`: A unaligned string.
///
/// # Returns
///
/// A vector of edits to transform thenaligned version of `x` into the aligned version of `y`.
#[must_use]
pub fn aligned_x_to_y(x: &str, y: &str) -> Vec<Edit> {
    let table = compute_table::<u16>(x, y, Penalties::default());
    let (aligned_x, aligned_y) = trace_back_iterative(&table, [x, y]);
    let mut aligned_x_to_y: Vec<Edit> = Vec::new();
    let mut modifier = 0;

    aligned_x
        .chars()
        .zip(aligned_y.chars())
        .enumerate()
        .filter(|(_, (aligned_x, aligned_y))| aligned_x != aligned_y)
        .for_each(|(index, (c_x, c_y))| {
            let i = index - modifier;

            if c_x == '-' {
                aligned_x_to_y.push(Edit::Ins(i, c_y));
            } else if c_y == '-' {
                aligned_x_to_y.push(Edit::Del(i));
                modifier += 1;
            } else {
                aligned_x_to_y.push(Edit::Sub(i, c_y));
            }
        });
    aligned_x_to_y
}

/// Given two unaligned strings, returns the edits related to gaps to align the 2 strings.
///
/// Requires that gaps are never aligned with gaps (our NW implementation with the default penalties ensures this).
/// Uses the `traceback_iterative` ftn to align the strings.
///
/// # Arguments
///
/// * `x`: A unaligned string.
/// * `y`: A unaligned string.
///
/// # Returns
///
/// A vector of edits to transform the aligned version of `x` into the aligned version of `y` excluding substitutions.
#[must_use]
pub fn aligned_x_to_y_no_sub(x: &str, y: &str) -> Vec<Edit> {
    let table = compute_table::<u16>(x, y, Penalties::default());
    let (aligned_x, aligned_y) = trace_back_iterative(&table, [x, y]);
    let mut aligned_x_to_y: Vec<Edit> = Vec::new();
    let mut modifier = 0;
    aligned_x
        .chars()
        .zip(aligned_y.chars())
        .enumerate()
        .filter(|(_, (aligned_x, aligned_y))| aligned_x != aligned_y)
        .for_each(|(index, (c_x, c_y))| {
            let i = index - modifier;

            if c_x == '-' {
                aligned_x_to_y.push(Edit::Ins(i, c_y));
            } else if c_y == '-' {
                aligned_x_to_y.push(Edit::Del(i));
                modifier += 1;
            }
        });
    aligned_x_to_y
}

/// Given two unaligned strings, returns the location of the gaps needed to align the 2 strings.
///
/// Requires that gaps are never aligned with gaps (our NW implementation with the default penalties ensures this).
/// Uses the `traceback_iterative` ftn to align the strings.
///
/// # Arguments
///
/// * `x`: A unaligned string.
/// * `y`: A unaligned string.
///
/// # Returns
///
/// An array of 2 vectors of gaps to align `x` and `y`.
#[must_use]
pub fn x_to_y_alignment(x: &str, y: &str) -> [Vec<usize>; 2] {
    let table = compute_table::<u16>(x, y, Penalties::default());
    let (aligned_x, aligned_y) = trace_back_iterative(&table, [x, y]);
    let mut gap_indices: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    let mut modifier: usize = 0;
    aligned_x
        .chars()
        .zip(aligned_y.chars())
        .enumerate()
        .filter(|(_, (aligned_x, aligned_y))| aligned_x != aligned_y)
        .for_each(|(index, (c_x, c_y))| {
            let i = index - modifier;
            if c_x == '-' {
                gap_indices[0].push(index);
            } else if c_y == '-' {
                gap_indices[1].push(i);
                modifier += 1;
            }
        });
    gap_indices
}

/// Applies a set of edits to a reference (unaligned) string to get a target (unaligned) string.
///
/// # Arguments
///
/// * `x`: The unaligned reference string.
/// * `edits`: The edits to apply to the reference string.
///
/// # Returns
///
/// The unaligned target string.
#[must_use]
pub fn apply_edits(x: &str, edits: &[Edit]) -> String {
    let mut x: Vec<char> = x.chars().collect();

    for edit in edits {
        match edit {
            Edit::Sub(i, c) => {
                x[*i] = *c;
            }
            Edit::Ins(i, c) => {
                x.insert(*i, *c);
            }
            Edit::Del(i) => {
                x.remove(*i);
            }
        }
    }
    x.into_iter().collect()
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
