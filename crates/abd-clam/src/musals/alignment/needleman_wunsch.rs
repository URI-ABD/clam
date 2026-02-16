//! Needleman-Wunsch global sequence alignment algorithm.\

use crate::DistanceValue;

use super::{CostMatrix, Direction, Edit, Edits};

/// A table of edit distances between prefixes of two sequences, along with the direction of the optimal edit operation at each cell.
type DpTable<T> = Vec<Vec<(T, Direction)>>;

impl<T: DistanceValue> CostMatrix<T> {
    /// Compute the dynamic programming matrix for the Needleman-Wunsch global sequence alignment algorithm.
    pub fn nw_table<S1: AsRef<[char]>, S2: AsRef<[char]>>(&self, seq1: &S1, seq2: &S2) -> DpTable<T> {
        /* TODO(Najib)
        Consider a parallel implementation where cells on the same anti-diagonal are computed in parallel.
        Such an implementation would require the the previous two anti-diagonals to compute the current anti-diagonal.
        */

        let seq1 = seq1.as_ref();
        let seq2 = seq2.as_ref();
        let mut table: DpTable<T> = vec![vec![(T::zero(), Direction::Diagonal); seq1.len() + 1]; seq2.len() + 1];

        // Initialize the first row to the cost of inserting characters from the first sequence.
        for i in 1..table[0].len() {
            let cost = table[0][i - 1].0 + self.gap_ext_cost();
            table[0][i] = (cost, Direction::Left);
        }

        // Initialize the first column to the cost of inserting characters from the second sequence.
        for i in 1..table.len() {
            let cost = table[i - 1][0].0 + self.gap_ext_cost();
            table[i][0] = (cost, Direction::Up);
        }

        // Fill in the rest of the table.
        // On iteration (i, j), we will fill in the cell at (i + 1, j + 1).
        for (i, &yc) in seq2.iter().enumerate() {
            for (j, &xc) in seq1.iter().enumerate() {
                // Compute the costs of the three possible operations.

                // The cost of a substitution (or match).
                let diag_cost = table[i][j].0 + self.sub_cost(xc, yc);

                // The cost of inserting a character depends on the previous operation.
                let up_cost = table[i][j + 1].0
                    + match table[i][j + 1].1 {
                        Direction::Up => self.gap_ext_cost(),
                        _ => self.gap_open_cost(),
                    };
                let left_cost = table[i + 1][j].0
                    + match table[i + 1][j].1 {
                        Direction::Left => self.gap_ext_cost(),
                        _ => self.gap_open_cost(),
                    };

                // Choose the operation with the minimum cost.
                // If there are ties, prefer diagonal > up > left. This will produce the shortest aligned sequences.
                table[i + 1][j + 1] = if diag_cost <= up_cost && diag_cost <= left_cost {
                    (diag_cost, Direction::Diagonal)
                } else if up_cost <= left_cost {
                    (up_cost, Direction::Up)
                } else {
                    (left_cost, Direction::Left)
                };
            }
        }

        table
    }

    /// Compute the Needleman-Wunsch distance between two sequences.
    pub fn nw_distance<S1: AsRef<[char]>, S2: AsRef<[char]>>(&self, seq1: &S1, seq2: &S2) -> T {
        let table = self.nw_table(seq1, seq2);
        table.last().and_then(|row| row.last()).map_or_else(T::zero, |&(cost, _)| cost)
    }

    /// Compute the optimal edit script to transform `seq1` into `seq2`, and vice-versa, using the Needleman-Wunsch algorithm.
    pub fn nw_edit_scripts<S1: AsRef<[char]>, S2: AsRef<[char]>>(&self, seq1: &S1, seq2: &S2) -> [Edits; 2] {
        let table = self.nw_table(seq1, seq2);

        let seq1 = seq1.as_ref();
        let seq2 = seq2.as_ref();
        let (mut row_i, mut col_i) = (seq2.len(), seq1.len());
        let (mut x2y_edits, mut y2x_edits) = (Vec::new(), Vec::new());

        while row_i > 0 || col_i > 0 {
            let char_col = seq1[col_i - 1];
            let char_row = seq2[row_i - 1];

            match table[row_i][col_i].1 {
                Direction::Diagonal => {
                    if char_col != char_row {
                        x2y_edits.push((col_i - 1, Edit::Sub(char_row)));
                        y2x_edits.push((row_i - 1, Edit::Sub(char_col)));
                    }
                    row_i -= 1;
                    col_i -= 1;
                }
                Direction::Up => {
                    x2y_edits.push((col_i, Edit::Ins(char_row)));
                    y2x_edits.push((row_i - 1, Edit::Del));
                    row_i -= 1;
                }
                Direction::Left => {
                    x2y_edits.push((col_i - 1, Edit::Del));
                    y2x_edits.push((row_i, Edit::Ins(char_col)));
                    col_i -= 1;
                }
            }
        }

        x2y_edits.reverse();
        y2x_edits.reverse();

        [Edits(x2y_edits), Edits(y2x_edits)]
    }

    /// Compute the optimal indices where gaps should be inserted to align `self` and `other` to each other, using the
    /// Needleman-Wunsch algorithm.
    #[must_use]
    pub fn nw_gap_indices<S1: AsRef<[char]>, S2: AsRef<[char]>>(&self, seq1: &S1, seq2: &S2) -> [Vec<usize>; 2] {
        let table = self.nw_table(seq1, seq2);

        let seq1 = seq1.as_ref();
        let seq2 = seq2.as_ref();
        let (mut row_i, mut col_i) = (seq2.len(), seq1.len());
        let (mut x_gaps, mut y_gaps) = (Vec::new(), Vec::new());

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

        [x_gaps, y_gaps]
    }
}
