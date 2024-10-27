//! Multiple Sequence Alignment with CLAM

mod cluster;
mod needleman_wunsch;

use distances::number::IInt;

pub use cluster::{Alignable, Gaps, PartialMSA};
pub use needleman_wunsch::{CostMatrix, NeedlemanWunschAligner};

use crate::{cakes::OffBall, Cluster, Dataset};

/// A multiple sequence alignment (MSA) builder.
#[allow(clippy::module_name_repetitions)]
pub struct MsaBuilder<'a, T: AsRef<[u8]>, U: IInt> {
    /// The Needleman-Wunsch aligner.
    aligner: &'a NeedlemanWunschAligner<U>,
    /// The columns of the partial MSA.
    columns: Vec<Vec<u8>>,
    /// Just to satisfy the compiler.
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: AsRef<[u8]>, U: IInt> MsaBuilder<'a, T, U> {
    /// Create a new MSA builder.
    pub const fn new(aligner: &'a NeedlemanWunschAligner<U>) -> Self {
        Self {
            aligner,
            columns: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a binary tree of `Cluster`s to the MSA.
    #[must_use]
    pub fn with_binary_tree<D: Dataset<T, U>, C: Cluster<T, U, D>>(self, c: &OffBall<T, U, D, C>, data: &D) -> Self {
        if c.children().is_empty() {
            self.with_cluster(c, data)
        } else {
            if c.children().len() != 2 {
                unreachable!("Binary tree has more than two children.");
            }
            let aligner = self.aligner;
            let left = c.children()[0].2.as_ref();
            let right = c.children()[1].2.as_ref();

            let l_msa = Self::new(aligner).with_binary_tree(left, data);
            let r_msa = Self::new(aligner).with_binary_tree(right, data);

            let l_center = left
                .indices()
                .position(|i| i == left.arg_center())
                .unwrap_or_else(|| unreachable!("Left center not found"));
            let r_center = right
                .indices()
                .position(|i| i == right.arg_center())
                .unwrap_or_else(|| unreachable!("Right center not found"));
            l_msa.merge(l_center, r_msa, r_center)
        }
    }

    /// Replaces all sequences in the MSA with the given sequence.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The sequence to add.
    #[must_use]
    pub fn with_sequence(mut self, sequence: &T) -> Self {
        self.columns = sequence.as_ref().iter().map(|&c| vec![c]).collect();
        self
    }

    /// Adds sequences from a `Cluster` to the MSA.
    #[must_use]
    pub fn with_cluster<D: Dataset<T, U>, C: Cluster<T, U, D>>(self, c: &C, data: &D) -> Self {
        let center = Self::new(self.aligner).with_sequence(data.get(c.arg_center()));
        c.indices()
            .filter(|&i| i != c.arg_center())
            .map(|i| data.get(i))
            .map(|s| Self::new(self.aligner).with_sequence(s))
            .fold(center, |center, other| center.merge(0, other, 0))
    }

    /// The number of sequences in the MSA.
    pub fn len(&self) -> usize {
        self.columns.first().map_or(0, Vec::len)
    }

    /// The number of columns in the MSA.
    ///
    /// If the MSA is empty, this will return 0.
    #[must_use]
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Whether the MSA is empty.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty() || self.columns.iter().all(Vec::is_empty)
    }

    /// Get the sequence at the given index.
    #[must_use]
    pub fn get_sequence(&self, index: usize) -> Vec<u8> {
        self.columns.iter().map(|col| col[index]).collect()
    }

    /// Merge two MSAs.
    #[must_use]
    pub fn merge(mut self, self_center: usize, mut other: Self, other_center: usize) -> Self {
        let x = self.get_sequence(self_center);
        let y = other.get_sequence(other_center);
        let (_, [x_to_y, y_to_x]) = self.aligner.gaps(&x, &y);

        for i in x_to_y {
            self.add_gap(i).unwrap_or_else(|e| unreachable!("{e}"));
        }

        for i in y_to_x {
            other.add_gap(i).unwrap_or_else(|e| unreachable!("{e}"));
        }

        if self.width() == other.width() {
            let aligner = self.aligner;
            let columns = self
                .columns
                .into_iter()
                .zip(other.columns)
                .map(|(mut x, mut y)| {
                    x.append(&mut y);
                    x
                })
                .collect();

            Self {
                aligner,
                columns,
                _phantom: std::marker::PhantomData,
            }
        } else {
            unreachable!("MSAs have different widths: {} vs {}", self.width(), other.width());
        }
    }

    /// Add a gap column to the MSA.
    ///
    /// # Arguments
    ///
    /// - index: The index at which to add the gap column.
    ///
    /// # Errors
    ///
    /// - If the MSA is empty.
    /// - If the index is greater than the number of columns.
    pub fn add_gap(&mut self, index: usize) -> Result<(), String> {
        if self.columns.is_empty() {
            Err("MSA is empty.".to_string())
        } else if index > self.columns.len() {
            Err("Index is greater than the number of columns.".to_string())
        } else {
            let gap_col = vec![b'-'; self.columns[0].len()];
            self.columns.insert(index, gap_col);
            Ok(())
        }
    }

    /// Convert the MSA builder into a multiple sequence alignment.
    #[must_use]
    pub fn as_msa(&self) -> Vec<Vec<u8>> {
        if self.is_empty() {
            Vec::new()
        } else {
            let mut rows = Vec::with_capacity(self.columns[0].len());
            for i in 0..self.columns[0].len() {
                rows.push(self.columns.iter().map(|col| col[i]).collect());
            }
            rows
        }
    }
}
