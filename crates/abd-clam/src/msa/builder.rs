//! Recursively build up the MSA using the CLAM tree.

use core::ops::Neg;
use std::string::FromUtf8Error;

use distances::Number;
use rayon::prelude::*;

use super::needleman_wunsch::Aligner;

use crate::{cakes::OffBall, cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

/// A multiple sequence alignment (MSA) builder.
pub struct Builder<'a, U: Number + Neg<Output = U>> {
    /// The Needleman-Wunsch aligner.
    pub(crate) aligner: &'a Aligner<'a, U>,
    /// The columns of the partial MSA.
    columns: Vec<Vec<u8>>,
}

impl<'a, U: Number + Neg<Output = U>> Builder<'a, U> {
    /// Create a new MSA builder.
    #[must_use]
    pub const fn new(aligner: &'a Aligner<U>) -> Self {
        Self {
            aligner,
            columns: Vec::new(),
        }
    }

    /// Get the gap character.
    #[must_use]
    pub const fn gap(&self) -> u8 {
        self.aligner.gap()
    }

    /// Add a binary tree of `Cluster`s to the MSA.
    #[must_use]
    pub fn with_binary_tree<T, D, C>(self, c: &OffBall<T, U, D, C>, data: &D) -> Self
    where
        T: AsRef<[u8]>,
        D: Dataset<T, U>,
        C: Cluster<T, U, D>,
    {
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
    pub fn with_sequence<T: AsRef<[u8]>>(mut self, sequence: &T) -> Self {
        self.columns = sequence.as_ref().iter().map(|&c| vec![c]).collect();
        self
    }

    /// Adds sequences from a `Cluster` to the MSA.
    #[must_use]
    pub fn with_cluster<T, D, C>(self, c: &C, data: &D) -> Self
    where
        T: AsRef<[u8]>,
        D: Dataset<T, U>,
        C: Cluster<T, U, D>,
    {
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

    /// Get the sequence at the given index.
    ///
    /// This is a convenience method that converts the sequence to a `String`.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the sequence to get.
    ///
    /// # Errors
    ///
    /// If the sequence is not valid UTF-8.
    pub fn get_sequence_str(&self, index: usize) -> Result<String, FromUtf8Error> {
        String::from_utf8(self.get_sequence(index))
    }

    /// Merge two MSAs.
    #[must_use]
    pub fn merge(mut self, self_center: usize, mut other: Self, other_center: usize) -> Self {
        let x = self.get_sequence(self_center);
        let y = other.get_sequence(other_center);
        let (_, [x_to_y, y_to_x]) = self.aligner.alignment_gaps(&x, &y);

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

            Self { aligner, columns }
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
        } else if index > self.width() {
            Err(format!(
                "Index is greater than the width of the MSA: {index} > {}",
                self.width()
            ))
        } else {
            let gap_col = vec![self.gap(); self.columns[0].len()];
            self.columns.insert(index, gap_col);
            Ok(())
        }
    }

    /// Extract the multiple sequence alignment.
    #[must_use]
    pub fn extract_msa(&self) -> Vec<Vec<u8>> {
        if self.is_empty() {
            Vec::new()
        } else {
            (0..self.len()).map(|i| self.get_sequence(i)).collect()
        }
    }

    /// Extract the multiple sequence alignment over `String`s.
    ///
    /// # Errors
    ///
    /// - If any of the sequences are not valid UTF-8.
    pub fn extract_msa_strings(&self) -> Result<Vec<String>, FromUtf8Error> {
        self.extract_msa().into_iter().map(String::from_utf8).collect()
    }
}

impl<'a, U: Number + Neg<Output = U>> Builder<'a, U> {
    /// Parallel version of `with_binary_tree`.
    #[must_use]
    pub fn par_with_binary_tree<T, D, C>(self, c: &OffBall<T, U, D, C>, data: &D) -> Self
    where
        T: AsRef<[u8]> + Send + Sync,
        D: ParDataset<T, U>,
        C: ParCluster<T, U, D>,
    {
        if c.children().is_empty() {
            self.with_cluster(c, data)
        } else {
            if c.children().len() != 2 {
                unreachable!("Binary tree has more than two children.");
            }
            let aligner = self.aligner;
            let left = c.children()[0].2.as_ref();
            let right = c.children()[1].2.as_ref();

            let (l_msa, r_msa) = rayon::join(
                || Self::new(aligner).par_with_binary_tree(left, data),
                || Self::new(aligner).par_with_binary_tree(right, data),
            );

            let l_center = left
                .indices()
                .position(|i| i == left.arg_center())
                .unwrap_or_else(|| unreachable!("Left center not found"));
            let r_center = right
                .indices()
                .position(|i| i == right.arg_center())
                .unwrap_or_else(|| unreachable!("Right center not found"));

            l_msa.par_merge(l_center, r_msa, r_center)
        }
    }

    /// Parallel version of `merge`.
    #[must_use]
    pub fn par_merge(mut self, self_center: usize, mut other: Self, other_center: usize) -> Self {
        let x = self.get_sequence(self_center);
        let y = other.get_sequence(other_center);
        let (_, [x_to_y, y_to_x]) = self.aligner.alignment_gaps(&x, &y);

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
                .into_par_iter()
                .zip(other.columns)
                .map(|(mut x, mut y)| {
                    x.append(&mut y);
                    x
                })
                .collect();

            Self { aligner, columns }
        } else {
            unreachable!("MSAs have different widths: {} vs {}", self.width(), other.width());
        }
    }

    /// Parallel version of `extract_msa`.
    #[must_use]
    pub fn par_extract_msa(&self) -> Vec<Vec<u8>> {
        if self.is_empty() {
            Vec::new()
        } else {
            (0..self.len()).into_par_iter().map(|i| self.get_sequence(i)).collect()
        }
    }

    /// Parallel version of `extract_msa_strings`.
    ///
    /// # Errors
    ///
    /// See `extract_msa_strings`.
    pub fn par_extract_msa_strings(&self) -> Result<Vec<String>, FromUtf8Error> {
        self.extract_msa().into_par_iter().map(String::from_utf8).collect()
    }
}
