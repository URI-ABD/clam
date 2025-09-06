//! A trait for sequence types used in `MuSAlS`.

use crate::DistanceValue;

use super::{
    CostMatrix,
    alignment_ops::{Direction, Edit, Edits},
};

/* TODO(Najib):
Add a new representation of an aligned sequence that stores the original sequence and in indices where the characters are to be inserted. Assuming that most of
the aligned sequences have many gaps, such a representation will save space when creating and storing aligned sequences.
*/

/// A table of edit distances between prefixes of two `Sequence`s.
type DpTable<T> = Vec<Vec<(T, Direction)>>;

/// Types that implement `Sequence` can be used in the `MuSAlS` algorithm.
///
/// Given some simple methods for creating and modifying sequences, the `Sequence` trait provides methods for using the Needleman-Wunsch algorithm to align
/// sequences and to compute the edit distances between them. It also provides methods for inserting gaps into sequences and for applying a series of edits to a
/// sequence to facilitate the alignment process.
///
/// # Examples
///
/// The `Sequence` trait can easily be implemented for types like `String` and `Vec<u8>`. Our work in the `MuSAlS` paper effectively uses the following
/// implementation:
///
/// ```rust
/// use abd_clam::musals::Sequence;
///
/// /// This is a simple wrapper around `String` so we can avoid the orphan rule.
/// #[derive(Clone, Default)]
/// struct MyString(String);
///
/// impl AsRef<[u8]> for MyString {
///     fn as_ref(&self) -> &[u8] {
///         self.0.as_bytes()
///     }
/// }
///
/// impl Sequence for MyString {
///    const GAP: u8 = b'-';
///
///    fn from_vec(v: Vec<u8>) -> Self {
///        Self(String::from_utf8(v).expect("Invalid UTF-8 sequence"))
///    }
///
///    fn from_string(s: String) -> Self {
///        Self(s)
///    }
///
///    fn append(&mut self, other: Self) {
///        self.0.push_str(&other.0);
///    }
///
///    fn pre_pend(&mut self, byte: u8) {
///        self.0.insert(0, byte as char);
///    }
///
///    fn post_pend(&mut self, byte: u8) {
///        self.0.push(byte as char);
///    }
/// }
/// ```
#[must_use]
pub trait Sequence: AsRef<[u8]> + Sized + Clone + Default {
    /// The gap character for the sequence type.
    const GAP: u8;

    /// Creates a sequence from a vector of bytes.
    #[must_use]
    fn from_vec(v: Vec<u8>) -> Self;

    /// Creates a sequence from a String.
    #[must_use]
    fn from_string(s: String) -> Self;

    /// Appends the other sequence to this sequence.
    fn append(&mut self, other: Self);

    /// Prepends a character to the sequence.
    fn pre_pend(&mut self, byte: u8);

    /// Appends a character to the sequence.
    fn post_pend(&mut self, byte: u8);

    /// Creates a sequence by copying a byte.
    #[must_use]
    fn splat(byte: u8, count: usize) -> Self {
        Self::from_vec(vec![byte; count])
    }

    /// Removes all gaps from the sequence.
    #[must_use]
    fn without_gaps(&self) -> Self {
        Self::from_vec(self.as_ref().iter().filter(|&&b| b != Self::GAP).copied().collect())
    }

    /// Removes all occurrences of the given bytes from the sequence.
    #[must_use]
    fn without_bytes(&self, bytes: &[u8]) -> Self {
        Self::from_vec(self.as_ref().iter().filter(|&&b| !bytes.contains(&b)).copied().collect())
    }

    /// Returns the number of gaps in the sequence.
    fn gap_count(&self) -> usize {
        bytecount::count(self.as_ref(), Self::GAP)
    }

    /// Computes the dynamic programming table using the Needleman-Wunsch algorithm and the given cost matrix.
    fn nw_table<T: DistanceValue>(&self, other: &Self, cost_matrix: &CostMatrix<T>) -> DpTable<T> {
        /* TODO(Najib)
        Consider a parallel implementation where cells on the same anti-diagonal are computed in parallel.
        Such an implementation would require the the previous two anti-diagonals to compute the current anti-diagonal.
        */

        let (x, y) = (self.as_ref(), other.as_ref());

        // Initialize the DP table.
        let mut table: DpTable<T> = vec![vec![(T::zero(), Direction::Diagonal); x.len() + 1]; y.len() + 1];

        // Initialize the first row to the cost of inserting characters from the first sequence.
        for i in 1..table[0].len() {
            let cost = table[0][i - 1].0 + cost_matrix.gap_ext_cost();
            table[0][i] = (cost, Direction::Left);
        }

        // Initialize the first column to the cost of inserting characters from the second sequence.
        for i in 1..table.len() {
            let cost = table[i - 1][0].0 + cost_matrix.gap_ext_cost();
            table[i][0] = (cost, Direction::Up);
        }

        // Fill in the rest of the table.
        // On iteration (i, j), we will fill in the cell at (i + 1, j + 1).
        for (i, &yc) in y.iter().enumerate() {
            for (j, &xc) in x.iter().enumerate() {
                // Compute the costs of the three possible operations.

                // The cost of a substitution (or match).
                let diag_cost = table[i][j].0 + cost_matrix.sub_cost(xc, yc);

                // The cost of inserting a character depends on the previous operation.
                let up_cost = table[i][j + 1].0
                    + match table[i][j + 1].1 {
                        Direction::Up => cost_matrix.gap_ext_cost(),
                        _ => cost_matrix.gap_open_cost(),
                    };
                let left_cost = table[i + 1][j].0
                    + match table[i + 1][j].1 {
                        Direction::Left => cost_matrix.gap_ext_cost(),
                        _ => cost_matrix.gap_open_cost(),
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

    /// Computes the Needleman-Wunsch dynamic programming table for aligning this sequence to another sequence without allowing gaps in the other sequence.
    ///
    /// # Errors
    ///
    /// If `self` is longer than `other`.
    fn nw_table_one_way<T: DistanceValue>(&self, other: &Self, cost_matrix: &CostMatrix<T>) -> Result<DpTable<T>, String> {
        // TODO: Consider a parallel implementation where cells on the same anti-diagonal are computed in parallel.

        let (x, y) = (self.as_ref(), other.as_ref());
        if x.len() > y.len() {
            return Err(format!("`self` must not be longer than `other`. Got lengths {} and {}", x.len(), y.len()));
        }

        // Initialize the DP table.
        let mut table: DpTable<T> = vec![vec![(T::zero(), Direction::Diagonal); x.len() + 1]; y.len() + 1];
        if x.len() == y.len() {
            return Ok(table);
        }

        // Initialize the first row to the cost of inserting characters from the first sequence.
        for i in 1..table[0].len() {
            let cost = table[0][i - 1].0 + cost_matrix.gap_ext_cost();
            table[0][i] = (cost, Direction::Left);
        }

        // Initialize the first column to the cost of inserting characters from the second sequence.
        for i in 1..table.len() {
            let cost = table[i - 1][0].0 + cost_matrix.gap_ext_cost();
            table[i][0] = (cost, Direction::Up);
        }

        // Fill in the rest of the table.
        // On iteration (i, j), we will fill in the cell at (i + 1, j + 1).
        for (i, &yc) in y.iter().enumerate() {
            for (j, &xc) in x.iter().enumerate() {
                // Compute the costs of the three possible operations.

                // The cost of a substitution (or match).
                let diag_cost = table[i][j].0 + cost_matrix.sub_cost(xc, yc);

                // The cost of inserting a character depends on the previous operation.
                let up_cost = table[i][j + 1].0
                    + match table[i][j + 1].1 {
                        Direction::Up => cost_matrix.gap_ext_cost(),
                        _ => cost_matrix.gap_open_cost(),
                    };

                // Choose the operation with the minimum cost.
                // If there are ties, prefer diagonal > up. This will produce the shortest aligned sequences.
                table[i + 1][j + 1] = if diag_cost <= up_cost {
                    (diag_cost, Direction::Diagonal)
                } else {
                    (up_cost, Direction::Up)
                };
            }
        }

        Ok(table)
    }

    /// Compute the Needleman-Wunsch distance between this sequence and another sequence.
    fn nw_distance<T: DistanceValue>(&self, other: &Self, cost_matrix: &CostMatrix<T>) -> T {
        let table = self.nw_table(other, cost_matrix);
        table.last().and_then(|row| row.last()).map_or_else(T::zero, |&(d, _)| d)
    }

    /// Applies the given edits to the sequence.
    #[must_use]
    fn apply_edits(&self, edits: &Edits) -> Self {
        let mut result = Vec::with_capacity(self.as_ref().len() + edits.len());
        result.extend(self.as_ref());

        let mut offset = 0;
        for (i, edit) in edits.iter() {
            match edit {
                Edit::Sub(c) => {
                    result[i + offset] = *c;
                }
                Edit::Ins(c) => {
                    result.insert(i + offset, *c);
                    offset += 1;
                }
                Edit::Del => {
                    result.remove(i + offset);
                    offset -= 1;
                }
            }
        }

        Self::from_vec(result)
    }

    /// Inserts gaps at the specified indices in the sequence.
    #[must_use]
    fn insert_gaps(&self, indices: &[usize]) -> Self {
        let mut result = Vec::with_capacity(self.as_ref().len() + indices.len());
        result.extend(self.as_ref());

        for &idx in indices.iter().rev() {
            result.insert(idx, Self::GAP);
        }

        Self::from_vec(result)
    }

    /// Returns the indices where gaps should be inserted to align the two sequences to each other.
    #[must_use]
    fn gap_indices<T: DistanceValue>(&self, other: &Self, dp_table: &DpTable<T>) -> [Vec<usize>; 2] {
        let (x, y) = (self.as_ref(), other.as_ref());
        let [mut row_i, mut col_i] = [y.len(), x.len()];
        let [mut x_gaps, mut y_gaps] = [Vec::new(), Vec::new()];

        while row_i > 0 || col_i > 0 {
            match dp_table[row_i][col_i].1 {
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

    /// Returns the indices where gaps should be inserted in this sequence to align it to another sequence.
    ///
    /// # Errors
    ///
    /// If the DP table contains a `Left` direction, which should not occur in one-way alignment. To avoid this, ensure
    /// that `self` is the shorter sequence and call `nw_table_one_way` to generate the DP table.
    fn gap_indices_one_way<T: DistanceValue>(&self, other: &Self, dp_table: &DpTable<T>) -> Result<Vec<usize>, String> {
        let (x, y) = (self.as_ref(), other.as_ref());
        let [mut row_i, mut col_i] = [y.len(), x.len()];
        let mut x_gaps = Vec::new();
        while row_i > 0 || col_i > 0 {
            match dp_table[row_i][col_i].1 {
                Direction::Diagonal => {
                    row_i -= 1;
                    col_i -= 1;
                }
                Direction::Up => {
                    x_gaps.push(col_i);
                    row_i -= 1;
                }
                Direction::Left => {
                    return Err(String::from("Left direction should not occur in one-way alignment"));
                }
            }
        }

        x_gaps.reverse();
        Ok(x_gaps)
    }
}

#[cfg(test)]
mod tests {
    use super::{CostMatrix, Edit, Edits, Sequence};

    impl Sequence for String {
        const GAP: u8 = b'-';

        #[allow(clippy::expect_used)]
        fn from_vec(v: Vec<u8>) -> Self {
            Self::from_utf8(v).expect("Invalid UTF-8 sequence")
        }

        fn from_string(s: String) -> Self {
            s
        }

        fn append(&mut self, other: Self) {
            self.push_str(&other);
        }

        fn pre_pend(&mut self, byte: u8) {
            self.insert(0, byte as char);
        }

        fn post_pend(&mut self, byte: u8) {
            self.push(byte as char);
        }
    }

    impl Sequence for Vec<u8> {
        const GAP: u8 = b'-';

        fn from_vec(vec: Vec<u8>) -> Self {
            vec
        }

        fn from_string(s: String) -> Self {
            s.into_bytes()
        }

        fn append(&mut self, other: Self) {
            self.extend_from_slice(&other);
        }

        fn pre_pend(&mut self, byte: u8) {
            self.insert(0, byte);
        }

        fn post_pend(&mut self, byte: u8) {
            self.push(byte);
        }
    }

    #[test]
    fn test_apply_edits() {
        fn test_generic<S: Sequence>(seq: &S) {
            let edits = Edits::new(vec![(1, Edit::Sub(b'T'))]);
            let mut edited_seq = seq.apply_edits(&edits);
            assert_eq!(edited_seq.as_ref(), b"ATGT");

            let edits = Edits::new(vec![(2, Edit::Ins(b'A'))]);
            edited_seq = edited_seq.apply_edits(&edits);
            assert_eq!(edited_seq.as_ref(), b"ATAGT");

            let edits = Edits::new(vec![(4, Edit::Del)]);
            edited_seq = edited_seq.apply_edits(&edits);
            assert_eq!(edited_seq.as_ref(), b"ATAG");

            let edits = Edits::new(vec![
                (1, Edit::Sub(b'T')), // This will change 'C' to 'T'
                (2, Edit::Ins(b'A')), // This will insert 'A' after the 'T' we just substituted
                (2, Edit::Del),       // This will delete the 'G' from the original sequence
            ]);
            let edited_seq = seq.apply_edits(&edits);
            assert_eq!(edited_seq.as_ref(), b"ATAT");
        }

        let seq = String::from("ACGT");
        test_generic(&seq);

        let seq = seq.as_bytes().to_vec();
        test_generic(&seq);
    }

    #[test]
    fn test_insert_gaps() {
        fn test_generic<S: Sequence>(seq: &S) {
            let gaps = vec![1, 3];
            let gapped_seq = seq.insert_gaps(&gaps);
            assert_eq!(gapped_seq.as_ref(), b"A-CG-T");
        }

        let seq = String::from("ACGT");
        test_generic(&seq);

        let seq = seq.as_bytes().to_vec();
        test_generic(&seq);
    }

    #[test]
    fn test_gap_indices() {
        fn test_generic<S: Sequence>(seq1: &S, seq2: &S, cost_matrix: &CostMatrix<i8>) {
            let dp_table = seq1.nw_table(seq2, cost_matrix);
            let [gaps1, gaps2] = seq1.gap_indices(seq2, &dp_table);
            assert_eq!(gaps1, vec![]);
            assert_eq!(gaps2, vec![1]);

            let gapped1 = seq1.insert_gaps(&gaps1);
            let gapped2 = seq2.insert_gaps(&gaps2);

            let gapped1 = gapped1.as_ref();
            let gapped2 = gapped2.as_ref();

            assert_eq!(gapped1.len(), gapped2.len());
            for (&c1, &c2) in gapped1.iter().zip(gapped2.iter()) {
                assert!((c1 == c2 && c1 != b'-') || c1 == b'-' || c2 == b'-');
            }
        }

        let cost_matrix = CostMatrix::default();

        let seq1 = String::from("ACGT");
        let seq2 = String::from("AGT");
        test_generic(&seq1, &seq2, &cost_matrix);

        let seq1 = seq1.as_bytes().to_vec();
        let seq2 = seq2.as_bytes().to_vec();
        test_generic(&seq1, &seq2, &cost_matrix);
    }

    #[test]
    fn test_gaps_one_way() -> Result<(), String> {
        fn test_generic<S: Sequence>(seq1: &S, seq2: &S, cost_matrix: &CostMatrix<i8>) -> Result<(), String> {
            let dp_table = seq1.nw_table_one_way(seq2, cost_matrix)?;
            let gaps1 = seq1.gap_indices_one_way(seq2, &dp_table)?;
            assert_eq!(gaps1, vec![1]);

            let gapped1 = seq1.insert_gaps(&gaps1);

            let gapped1 = gapped1.as_ref();
            let gapped2 = seq2.as_ref();

            assert_eq!(gapped1.len(), gapped2.len());
            for (&c1, &c2) in gapped1.iter().zip(gapped2.iter()) {
                assert!((c1 == c2 && c1 != b'-') || c1 == b'-' || c2 == b'-');
            }
            Ok(())
        }

        let cost_matrix = CostMatrix::default();

        let seq1 = String::from("AGT");
        let seq2 = String::from("ACGT");
        test_generic(&seq1, &seq2, &cost_matrix)?;

        let seq1 = seq1.as_bytes().to_vec();
        let seq2 = seq2.as_bytes().to_vec();
        test_generic(&seq1, &seq2, &cost_matrix)?;

        Ok(())
    }
}
