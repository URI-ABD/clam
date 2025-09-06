//! A multiple sequence alignment (MSA) of sequences of type `S`.

use rayon::prelude::*;

use crate::{Cluster, DistanceValue};

mod alignment_ops;
mod cost_matrix;
mod sequence;

pub use cost_matrix::CostMatrix;
pub use sequence::Sequence;

/// A multiple sequence alignment stored in a columnar format.
#[derive(Clone, Debug, Default)]
#[must_use]
pub struct PartialMsa<S: Sequence>(Vec<S>);

impl<S: Sequence> core::ops::Deref for PartialMsa<S> {
    type Target = Vec<S>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: Sequence> FromIterator<S> for PartialMsa<S> {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<S: Sequence + Send + Sync> FromParallelIterator<S> for PartialMsa<S> {
    fn from_par_iter<I: IntoParallelIterator<Item = S>>(par_iter: I) -> Self {
        Self(par_iter.into_par_iter().collect())
    }
}

impl<S: Sequence> IntoIterator for PartialMsa<S> {
    type Item = S;
    type IntoIter = std::vec::IntoIter<S>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<S: Sequence + Send + Sync> IntoParallelIterator for PartialMsa<S> {
    type Item = S;
    type Iter = rayon::vec::IntoIter<S>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.into_par_iter()
    }
}

/// A trait for types that can be represented in a columnar format.
impl<S: Sequence> PartialMsa<S> {
    /// Creates a new `PartialMsa` MSA for a leaf cluster.
    pub fn from_leaf<Id, T, A>(leaf: &Cluster<T, A>, leaf_items: &[(Id, S)], cost_matrix: &CostMatrix<T>) -> Self
    where
        T: DistanceValue,
    {
        ftlog::debug!("Aligning leaf cluster at depth {} with {} sequences", leaf.depth, leaf.cardinality);

        let mut items = leaf_items.iter().map(|(_, seq)| Self::from_row(seq)).collect::<Vec<_>>();
        let mut bottom = items.pop().unwrap_or_else(|| unreachable!("Leaf cluster is never empty"));
        while let Some(prev) = items.pop() {
            bottom = bottom.merge(prev, cost_matrix);
        }

        ftlog::debug!(
            "Finished aligning leaf cluster at depth {} with {} sequences to {} width",
            leaf.depth,
            leaf.cardinality,
            bottom.len()
        );

        bottom
    }

    /// Creates a new `PartialMsa` MSA for a parent cluster.
    pub fn from_parent<T>(parent_center: &S, mut child_alignments: Vec<Self>, cost_matrix: &CostMatrix<T>) -> Self
    where
        T: DistanceValue,
    {
        let mut bottom = child_alignments.pop().unwrap_or_else(|| unreachable!("Parent cluster is never empty"));
        while let Some(child) = child_alignments.pop() {
            bottom = bottom.merge(child, cost_matrix);
        }
        bottom.post_pend_row(parent_center, cost_matrix)
    }

    /// The number of sequences.
    #[must_use]
    pub fn n_seq(&self) -> usize {
        self[0].as_ref().len()
    }

    /// The length of each sequence.
    fn width(&self) -> usize {
        self.len()
    }

    /// Returns the last row as a representative sequence.
    fn last_row(&self) -> S {
        let h = self.n_seq() - 1;
        let mut last = vec![S::GAP; self.width()];
        for (c, col) in self.iter().enumerate() {
            last[c] = col.as_ref()[h];
        }
        S::from_vec(last)
    }

    /// Creates a new `PartialMsa` from a single sequence as a row.
    pub fn from_row(row: &S) -> Self {
        row.as_ref().iter().map(|&byte| S::splat(byte, 1)).collect()
    }

    /// Converts the columnar MSA into a vector of sequences (rows).
    pub fn into_rows(self, reverse: bool) -> Vec<S> {
        let n_seq = self.n_seq();
        let width = self.width();

        let mut rows = vec![vec![S::GAP; width]; n_seq];
        for (c, col) in self.into_iter().enumerate() {
            for (i, &byte) in col.as_ref().iter().enumerate() {
                rows[i][c] = byte;
            }
        }

        let rows_iter = rows.into_iter().map(S::from_vec);
        if reverse { rows_iter.rev().collect() } else { rows_iter.collect() }
    }

    /// Returns a column of all gaps.
    fn gap_column(&self) -> S {
        S::splat(S::GAP, self.n_seq())
    }

    /// Inserts gap columns at the specified positions.
    fn with_gaps(self, indices: &[usize]) -> Self {
        let gap_column = self.gap_column();

        let mut cols = self.into_iter().collect::<Vec<S>>();
        for &idx in indices.iter().rev() {
            cols.insert(idx, gap_column.clone());
        }

        Self(cols)
    }

    /// Merges another `PartialMsa` into this one.
    ///
    /// This method aligns the last row of both columnar structures using the Needleman-Wunsch algorithm and inserts gap-columns as necessary to align partial
    /// MSAs correctly. It then concatenates the respective columns.
    pub fn merge<T: DistanceValue>(self, top: Self, cost_matrix: &CostMatrix<T>) -> Self {
        let [b, t] = [self.last_row(), top.last_row()];
        let dp_table = b.nw_table(&t, cost_matrix);
        let [b_gaps, t_gaps] = b.gap_indices(&t, &dp_table);

        let top = top.with_gaps(&t_gaps);
        let bottom = self.with_gaps(&b_gaps);

        bottom
            .into_iter()
            .zip(top)
            .map(|(mut b_col, t_col)| {
                b_col.append(t_col);
                b_col
            })
            .collect()
    }

    /// Adds a new row to the bottom of the columnar structure.
    pub fn post_pend_row<T: DistanceValue>(self, row: &S, cost_matrix: &CostMatrix<T>) -> Self {
        let last = self.last_row();
        let dp_table = last.nw_table(row, cost_matrix);
        let [last_gaps, row_gaps] = last.gap_indices(row, &dp_table);
        let row = row.insert_gaps(&row_gaps);

        self.with_gaps(&last_gaps)
            .into_iter()
            .zip(row.as_ref())
            .map(|(mut col, &byte)| {
                col.post_pend(byte);
                col
            })
            .collect()
    }
}

impl<S: Sequence + Send + Sync> PartialMsa<S> {
    /// Parallel version of [`Self::from_leaf`].
    pub fn par_from_leaf<Id, T, A>(leaf: &Cluster<T, A>, leaf_items: &[(Id, S)], cost_matrix: &CostMatrix<T>) -> Self
    where
        Id: Send + Sync,
        T: DistanceValue + Send + Sync,
    {
        ftlog::debug!("Aligning leaf cluster at depth {} with {} sequences", leaf.depth, leaf.cardinality);

        let mut items = leaf_items.par_iter().map(|(_, seq)| Self::from_row(seq)).collect::<Vec<_>>();
        let mut bottom = items.pop().unwrap_or_else(|| unreachable!("Leaf cluster is never empty"));
        while let Some(prev) = items.pop() {
            bottom = bottom.par_merge(prev, cost_matrix);
        }

        ftlog::debug!(
            "Finished aligning leaf cluster at depth {} with {} sequences to {} width",
            leaf.depth,
            leaf.cardinality,
            bottom.len()
        );

        bottom
    }

    /// Parallel version of [`Self::from_parent`].
    pub fn par_from_parent<T>(parent_center: &S, mut child_alignments: Vec<Self>, cost_matrix: &CostMatrix<T>) -> Self
    where
        T: DistanceValue + Send + Sync,
    {
        let mut bottom = child_alignments.pop().unwrap_or_else(|| unreachable!("Parent cluster is never empty"));
        while let Some(child) = child_alignments.pop() {
            bottom = bottom.par_merge(child, cost_matrix);
        }
        bottom.post_pend_row(parent_center, cost_matrix)
    }

    /// Parallel version of [`Self::into_rows`].
    pub fn par_into_rows(self, reverse: bool) -> Vec<S> {
        let n_seq = self.n_seq();
        let width = self.width();

        let rows = vec![vec![S::GAP; width]; n_seq];
        self.into_par_iter().enumerate().for_each(|(c, col)| {
            col.as_ref().par_iter().enumerate().for_each(|(i, &byte)| {
                // SAFETY: We have exclusive access to each cell in the rows matrix
                // because every (c, i) pair is unique.
                #[allow(unsafe_code)]
                unsafe {
                    let row_ptr = &mut *rows.as_ptr().cast_mut().add(i);
                    row_ptr[c] = byte;
                }
            });
        });

        let rows_iter = rows.into_iter().map(S::from_vec);
        if reverse { rows_iter.rev().collect() } else { rows_iter.collect() }
    }

    /// Parallel version of [`Self::merge`].
    pub fn par_merge<T: DistanceValue + Send + Sync>(self, top: Self, cost_matrix: &CostMatrix<T>) -> Self {
        let [b, t] = [self.last_row(), top.last_row()];
        let dp_table = b.nw_table(&t, cost_matrix);
        let [b_gaps, t_gaps] = b.gap_indices(&t, &dp_table);

        let top = top.with_gaps(&t_gaps);
        let bottom = self.with_gaps(&b_gaps);

        bottom
            .into_par_iter()
            .zip(top.into_par_iter())
            .map(|(mut b_col, t_col)| {
                b_col.append(t_col);
                b_col
            })
            .collect()
    }
}
