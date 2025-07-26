//! Recursively build up the MSA using the CLAM tree.

use core::ops::Index;

use std::string::FromUtf8Error;

use distances::Number;
use rayon::prelude::*;

use crate::{cakes::PermutedBall, cluster::ParCluster, dataset::ParDataset, Cluster, Dataset, FlatVec};

use super::super::Aligner;

/// The columns of a partial MSA.
#[must_use]
pub struct Columns(Vec<Vec<u8>>, u8);

impl Index<usize> for Columns {
    type Output = Vec<u8>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Columns {
    /// Create a new MSA builder.
    ///
    /// # Arguments
    ///
    /// * `gap` - The character to use for the gap.
    pub const fn new(gap: u8) -> Self {
        Self(Vec::new(), gap)
    }

    /// Add a binary tree of `Cluster`s to the MSA.
    pub fn with_binary_tree<I, T, D, C>(self, c: &PermutedBall<T, C>, data: &D, aligner: &Aligner<T>) -> Self
    where
        I: AsRef<[u8]>,
        T: Number,
        D: Dataset<I>,
        C: Cluster<T>,
    {
        if c.children().is_empty() {
            self.with_cluster(c, data, aligner)
        } else {
            if c.children().len() != 2 {
                unreachable!("Binary tree has more than two children.");
            }
            let left = c.children()[0];
            let right = c.children()[1];

            let l_msa = Self::new(self.1).with_binary_tree(left, data, aligner);
            let r_msa = Self::new(self.1).with_binary_tree(right, data, aligner);

            let l_center = left
                .iter_indices()
                .position(|i| i == left.arg_center())
                .unwrap_or_else(|| unreachable!("Left center not found"));
            let r_center = right
                .iter_indices()
                .position(|i| i == right.arg_center())
                .unwrap_or_else(|| unreachable!("Right center not found"));

            l_msa.merge(l_center, r_msa, r_center, aligner)
        }
    }

    /// Add a tree of `Cluster`s to the MSA.
    pub fn with_tree<I, T, D, C>(self, c: &PermutedBall<T, C>, data: &D, aligner: &Aligner<T>) -> Self
    where
        I: AsRef<[u8]>,
        T: Number,
        D: Dataset<I>,
        C: Cluster<T>,
    {
        if c.children().is_empty() {
            self.with_cluster(c, data, aligner)
        } else {
            let children = c.children();
            let (&first, rest) = children.split_first().unwrap_or_else(|| unreachable!("No children"));

            let f_center = first
                .iter_indices()
                .position(|i| i == first.arg_center())
                .unwrap_or_else(|| unreachable!("First center not found"));
            let first = Self::new(self.1).with_tree(first, data, aligner);

            let (_, merged) = rest
                .iter()
                .map(|&o| {
                    let o_center = o
                        .iter_indices()
                        .position(|i| i == o.arg_center())
                        .unwrap_or_else(|| unreachable!("Other center not found"));
                    (o_center, Self::new(self.1).with_tree(o, data, aligner))
                })
                .fold((f_center, first), |(a_center, acc), (o_center, o)| {
                    (a_center, acc.merge(a_center, o, o_center, aligner))
                });

            merged
        }
    }

    /// Replaces all sequences in the MSA with the given sequence.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The sequence to add.
    pub fn with_sequence<I: AsRef<[u8]>>(mut self, sequence: &I) -> Self {
        self.0 = sequence.as_ref().iter().map(|&c| vec![c]).collect();
        self
    }

    /// Adds sequences from a `Cluster` to the MSA.
    pub fn with_cluster<I, T, D, C>(self, c: &C, data: &D, aligner: &Aligner<T>) -> Self
    where
        I: AsRef<[u8]>,
        T: Number,
        D: Dataset<I>,
        C: Cluster<T>,
    {
        ftlog::trace!(
            "Adding cluster to MSA. Depth: {}, Cardinality: {}",
            c.depth(),
            c.cardinality()
        );
        let indices = c.indices();
        let (&first, rest) = indices.split_first().unwrap_or_else(|| unreachable!("No indices"));
        let first = Self::new(self.1).with_sequence(data.get(first));
        rest.iter()
            .map(|&i| data.get(i))
            .map(|s| Self::new(self.1).with_sequence(s))
            .fold(first, |acc, s| acc.merge(0, s, 0, aligner))
    }

    /// The number of sequences in the MSA.
    pub fn len(&self) -> usize {
        self.0.first().map_or(0, Vec::len)
    }

    /// The number of columns in the MSA.
    ///
    /// If the MSA is empty, this will return 0.
    #[must_use]
    pub fn width(&self) -> usize {
        self.0.len()
    }

    /// Whether the MSA is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty() || self.0.iter().all(Vec::is_empty)
    }

    /// Get the columns of the MSA.
    #[must_use]
    pub const fn columns(&self) -> &Vec<Vec<u8>> {
        &self.0
    }

    /// Get the sequence at the given index.
    #[must_use]
    pub fn get_sequence(&self, index: usize) -> Vec<u8> {
        self.0.iter().map(|col| col[index]).collect()
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
    pub fn merge<T: Number>(mut self, s_center: usize, mut other: Self, o_center: usize, aligner: &Aligner<T>) -> Self {
        ftlog::trace!(
            "Merging MSAs with cardinalities: {} and {}, and centers {s_center} and {o_center}",
            self.len(),
            other.len()
        );
        let s_center = self.get_sequence(s_center);
        let o_center = other.get_sequence(o_center);

        let table = aligner.dp_table(&s_center, &o_center);
        let [s_to_o, o_to_s] = aligner.alignment_gaps(&s_center, &o_center, &table);

        for i in s_to_o {
            self.add_gap(i).unwrap_or_else(|e| unreachable!("{e}"));
        }

        for i in o_to_s {
            other.add_gap(i).unwrap_or_else(|e| unreachable!("{e}"));
        }

        let columns = self
            .0
            .into_iter()
            .zip(other.0)
            .map(|(mut x, mut y)| {
                x.append(&mut y);
                x
            })
            .collect();

        Self(columns, self.1)
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
        if self.0.is_empty() {
            Err("MSA is empty.".to_string())
        } else if index > self.width() {
            Err(format!(
                "Index is greater than the width of the MSA: {index} > {}",
                self.width()
            ))
        } else {
            let gap_col = vec![self.1; self.0[0].len()];
            self.0.insert(index, gap_col);
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

    /// Extract the columns as a `FlatVec`.
    pub fn to_flat_vec_columns(&self) -> FlatVec<Vec<u8>, usize> {
        FlatVec::new(self.0.clone())
            .unwrap_or_else(|e| unreachable!("{e}"))
            .with_dim_lower_bound(self.len())
            .with_dim_upper_bound(self.len())
            .with_name("ColWiseMSA")
    }

    /// Extract the rows as a `FlatVec`.
    pub fn to_flat_vec_rows(&self) -> FlatVec<Vec<u8>, usize> {
        FlatVec::new(self.extract_msa())
            .unwrap_or_else(|e| unreachable!("{e}"))
            .with_dim_lower_bound(self.width())
            .with_dim_upper_bound(self.width())
            .with_name("RowWiseMSA")
    }
}

impl Columns {
    /// Parallel version of [`Columnar::with_binary_tree`](crate::msa::dataset::columnar::Columnar::with_binary_tree).
    pub fn par_with_binary_tree<I, T, D, C>(self, c: &PermutedBall<T, C>, data: &D, aligner: &Aligner<T>) -> Self
    where
        I: AsRef<[u8]> + Send + Sync,
        T: Number,
        D: ParDataset<I>,
        C: ParCluster<T>,
    {
        if c.children().is_empty() {
            self.with_cluster(c, data, aligner)
        } else {
            if c.children().len() != 2 {
                unreachable!("Binary tree has more than two children.");
            }
            let left = c.children()[0];
            let right = c.children()[1];

            let (l_msa, r_msa) = rayon::join(
                || Self::new(self.1).par_with_binary_tree(left, data, aligner),
                || Self::new(self.1).par_with_binary_tree(right, data, aligner),
            );

            let l_center = left
                .iter_indices()
                .position(|i| i == left.arg_center())
                .unwrap_or_else(|| unreachable!("Left center not found"));
            let r_center = right
                .iter_indices()
                .position(|i| i == right.arg_center())
                .unwrap_or_else(|| unreachable!("Right center not found"));

            l_msa.par_merge(l_center, r_msa, r_center, aligner)
        }
    }

    /// Parallel version of [`Columnar::with_tree`](crate::msa::dataset::columnar::Columnar::with_tree).
    pub fn par_with_tree<I, T, D, C>(self, c: &PermutedBall<T, C>, data: &D, aligner: &Aligner<T>) -> Self
    where
        I: AsRef<[u8]> + Send + Sync,
        T: Number,
        D: ParDataset<I>,
        C: ParCluster<T>,
    {
        if c.children().is_empty() {
            self.with_cluster(c, data, aligner)
        } else {
            let children = c.children();
            let (&first, rest) = children.split_first().unwrap_or_else(|| unreachable!("No children"));

            let f_center = first
                .iter_indices()
                .position(|i| i == first.arg_center())
                .unwrap_or_else(|| unreachable!("First center not found"));
            let first = Self::new(self.1).with_tree(first, data, aligner);

            let (_, merged) = rest
                .par_iter()
                .map(|&o| {
                    let o_center = o
                        .iter_indices()
                        .position(|i| i == o.arg_center())
                        .unwrap_or_else(|| unreachable!("Other center not found"));
                    (o_center, Self::new(self.1).with_tree(o, data, aligner))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .fold((f_center, first), |(a_center, acc), (o_center, o)| {
                    (a_center, acc.par_merge(a_center, o, o_center, aligner))
                });

            merged
        }
    }

    /// Parallel version of [`Columnar::merge`](crate::msa::dataset::columnar::Columnar::merge).
    pub fn par_merge<T: Number>(
        mut self,
        s_center: usize,
        mut other: Self,
        o_center: usize,
        aligner: &Aligner<T>,
    ) -> Self {
        ftlog::trace!(
            "Parallel Merging MSAs with cardinalities: {} and {}, and centers {s_center} and {o_center}",
            self.len(),
            other.len()
        );
        let s_center = self.get_sequence(s_center);
        let o_center = other.get_sequence(o_center);
        let table = aligner.dp_table(&s_center, &o_center);
        let [s_to_o, o_to_s] = aligner.alignment_gaps(&s_center, &o_center, &table);

        for i in s_to_o {
            self.add_gap(i).unwrap_or_else(|e| unreachable!("{e}"));
        }

        for i in o_to_s {
            other.add_gap(i).unwrap_or_else(|e| unreachable!("{e}"));
        }

        let columns = self
            .0
            .into_par_iter()
            .zip(other.0)
            .map(|(mut x, mut y)| {
                x.append(&mut y);
                x
            })
            .collect();

        Self(columns, self.1)
    }

    /// Parallel version of [`Columnar::extract_msa`](crate::msa::dataset::columnar::Columnar::extract_msa).
    #[must_use]
    pub fn par_extract_msa(&self) -> Vec<Vec<u8>> {
        if self.is_empty() {
            Vec::new()
        } else {
            (0..self.len()).into_par_iter().map(|i| self.get_sequence(i)).collect()
        }
    }

    /// Parallel version of [`Columnar::extract_msa_strings`](crate::msa::dataset::columnar::Columnar::extract_msa_strings).
    ///
    /// # Errors
    ///
    /// See [`Columnar::extract_msa_strings`](crate::msa::dataset::columnar::Columnar::extract_msa_strings).
    pub fn par_extract_msa_strings(&self) -> Result<Vec<String>, FromUtf8Error> {
        self.extract_msa().into_par_iter().map(String::from_utf8).collect()
    }
}
