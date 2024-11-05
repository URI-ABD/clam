//! The Needleman-Wunsch aligner.

use core::ops::Neg;

use distances::Number;

use super::{CostMatrix, Direction, Edit, Edits};

/// A table of edit distances between prefixes of two sequences.
type NwTable<T> = Vec<Vec<(T, Direction)>>;

/// A Needleman-Wunsch aligner.
///
/// This works with any sequence of bytes, and also provides helpers for working
/// with strings.
#[derive(Clone)]
pub struct Aligner<'a, T: Number + Neg<Output = T>> {
    /// The cost matrix for the alignment.
    matrix: &'a CostMatrix<T>,
    /// The gap character.
    gap: u8,
    /// Whether to minimize the distance or maximize the similarity.
    minimizer: bool,
}

impl<'a, T: Number + Neg<Output = T>> Aligner<'a, T> {
    /// Create a new Needleman-Wunsch aligner that minimizes the cost.
    pub const fn new_minimizer(matrix: &'a CostMatrix<T>, gap: u8) -> Self {
        Self {
            matrix,
            gap,
            minimizer: true,
        }
    }

    /// Create a new Needleman-Wunsch aligner that minimizes the cost.
    pub const fn new_maximizer(matrix: &'a CostMatrix<T>, gap: u8) -> Self {
        Self {
            matrix,
            gap,
            minimizer: false,
        }
    }

    /// Get the gap character.
    #[must_use]
    pub const fn gap(&self) -> u8 {
        self.gap
    }

    /// Compute the minimized edit distance between two sequences' DP table.
    pub fn distance(&self, dp_table: &NwTable<T>) -> T {
        let d = dp_table.last().and_then(|row| row.last()).map_or(T::ZERO, |(d, _)| *d);
        if self.minimizer {
            d
        } else {
            -d
        }
    }

    /// Compute the dynamic programming table for the Needleman-Wunsch algorithm.
    ///
    /// The DP table is a 2D array of edit distances between prefixes of the two
    /// sequences. The value at position `(i, j)` is the edit distance between
    /// the first `i` characters of the first sequence and the first `j`
    /// characters of the second sequence.
    ///
    /// This implementation will minimize the edit distance.
    ///
    /// # Arguments
    ///
    /// * `x` - The first sequence.
    /// * `y` - The second sequence.
    ///
    /// # Returns
    ///
    /// The DP table.
    pub fn dp_table<S: AsRef<[u8]>>(&self, x: &S, y: &S) -> NwTable<T> {
        let (x, y) = (x.as_ref(), y.as_ref());

        // Initialize the DP table.
        let mut dp = vec![vec![(T::ZERO, Direction::Diagonal); x.len() + 1]; y.len() + 1];

        // The top-left cell is the cost of aligning two empty sequences.
        dp[0][0] = (T::ZERO, Direction::Diagonal);

        // Initialize the first row to the cost of inserting characters from the
        // first sequence.
        for i in 1..dp[0].len() {
            let cost = dp[0][i - 1].0 + self.matrix.gap_ext_cost();
            dp[0][i] = (cost, Direction::Left);
        }

        // Initialize the first column to the cost of inserting characters from
        // the second sequence.
        for j in 1..dp.len() {
            let cost = dp[j - 1][0].0 + self.matrix.gap_ext_cost();
            dp[j][0] = (cost, Direction::Up);
        }

        // Fill in the DP table.
        for (i, &yc) in y.iter().enumerate() {
            for (j, &xc) in x.iter().enumerate() {
                // The cost of a mismatch is the substitution cost.
                let mismatch_penalty = if yc == xc {
                    T::ZERO
                } else {
                    self.matrix.sub_cost(xc, yc)
                };

                // Compute the costs of the three possible operations.
                let diag_cost = dp[i][j].0 + mismatch_penalty;

                // The cost of inserting a character depends on the previous
                // operation.
                let up_cost = dp[i][j + 1].0
                    + match dp[i][j + 1].1 {
                        Direction::Up => self.matrix.gap_ext_cost(),
                        _ => self.matrix.gap_open_cost(),
                    };
                let left_cost = dp[i + 1][j].0
                    + match dp[i + 1][j].1 {
                        Direction::Left => self.matrix.gap_ext_cost(),
                        _ => self.matrix.gap_open_cost(),
                    };

                // Choose the operation with the minimum cost. If there is a tie,
                // prefer the diagonal operation so that the aligned sequences
                // are as short as possible.
                dp[i + 1][j + 1] = if diag_cost <= up_cost && diag_cost <= left_cost {
                    (diag_cost, Direction::Diagonal)
                } else if up_cost <= left_cost {
                    (up_cost, Direction::Up)
                } else {
                    (left_cost, Direction::Left)
                };
            }
        }

        dp
    }

    /// Align two sequences using the Needleman-Wunsch algorithm.
    ///
    /// # Arguments
    ///
    /// * `x` - The first sequence.
    /// * `y` - The second sequence.
    ///
    /// # Returns
    ///
    /// The alignment distance and the aligned sequences as bytes.
    pub fn align<S: AsRef<[u8]>>(&self, x: &S, y: &S, table: &NwTable<T>) -> (T, [Vec<u8>; 2]) {
        let (x, y) = (x.as_ref(), y.as_ref());
        let [mut row_i, mut col_i] = [y.len(), x.len()];
        let [mut x_aligned, mut y_aligned] = [
            Vec::with_capacity(x.len() + y.len()),
            Vec::with_capacity(x.len() + y.len()),
        ];

        while row_i > 0 || col_i > 0 {
            match table[row_i][col_i].1 {
                Direction::Diagonal => {
                    x_aligned.push(x[col_i - 1]);
                    y_aligned.push(y[row_i - 1]);
                    row_i -= 1;
                    col_i -= 1;
                }
                Direction::Up => {
                    x_aligned.push(self.gap);
                    y_aligned.push(y[row_i - 1]);
                    row_i -= 1;
                }
                Direction::Left => {
                    x_aligned.push(x[col_i - 1]);
                    y_aligned.push(self.gap);
                    col_i -= 1;
                }
            }
        }

        x_aligned.reverse();
        y_aligned.reverse();

        (self.distance(table), [x_aligned, y_aligned])
    }

    /// Align two strings using the Needleman-Wunsch algorithm.
    pub fn align_str<S: AsRef<str>>(&self, x: &S, y: &S, table: &NwTable<T>) -> (T, [String; 2]) {
        let (d, [x_aligned, y_aligned]) = self.align(&x.as_ref(), &y.as_ref(), table);
        (
            d,
            [
                String::from_utf8(x_aligned).unwrap_or_else(|e| unreachable!("We only added gaps: {e}")),
                String::from_utf8(y_aligned).unwrap_or_else(|e| unreachable!("We only added gaps: {e}")),
            ],
        )
    }

    /// Returns the `Edits` needed to align two sequences.
    ///
    /// Both sequences will need their respective `Edits` applied to them before
    /// they are in alignment.
    ///
    /// # Arguments
    ///
    /// * `x` - The first sequence.
    /// * `y` - The second sequence.
    ///
    /// # Returns
    ///
    /// The `Edits` needed to align the two sequences.
    pub fn edits<S: AsRef<[u8]>>(&self, x: &S, y: &S) -> (T, [Edits; 2]) {
        let table = self.dp_table(x, y);
        let (d, [x_aligned, y_aligned]) = self.align(x, y, &table);
        (
            d,
            [
                aligned_x_to_y(&x_aligned, &y_aligned),
                aligned_x_to_y(&y_aligned, &x_aligned),
            ],
        )
    }

    /// Returns the indices where gaps need to be inserted to align two
    /// sequences.
    pub fn alignment_gaps<S: AsRef<[u8]>>(&self, x: &S, y: &S) -> (T, [Vec<usize>; 2]) {
        let table = self.dp_table(x, y);

        let (x, y) = (x.as_ref(), y.as_ref());

        let [mut row_i, mut col_i] = [y.len(), x.len()];
        let [mut x_gaps, mut y_gaps] = [Vec::new(), Vec::new()];

        while row_i > 0 || col_i > 0 {
            match table[row_i][col_i].1 {
                Direction::Diagonal => {
                    row_i -= 1;
                    col_i -= 1;
                }
                Direction::Up => {
                    x_gaps.push(col_i);
                    row_i -= 1;
                }
                Direction::Left => {
                    y_gaps.push(row_i);
                    col_i -= 1;
                }
            }
        }

        x_gaps.reverse();
        y_gaps.reverse();

        (self.distance(&table), [x_gaps, y_gaps])
    }
}

/// A helper function to create a sequence of `Edits` needed to align one
/// sequence to another.
fn aligned_x_to_y<S: AsRef<[u8]>>(x: &S, y: &S) -> Edits {
    x.as_ref()
        .iter()
        .zip(y.as_ref().iter())
        .enumerate()
        .filter(|(_, (&xc, &yc))| xc != yc)
        .map(|(i, (_, &c))| (i, Edit::Sub(c)))
        .collect()
}
