//! Recursively build up the MSA using the CLAM tree.

use core::ops::Index;

use std::string::FromUtf8Error;

use distances::Number;
use rayon::prelude::*;

use crate::{cakes::PermutedBall, cluster::ParCluster, dataset::ParDataset, Cluster, Dataset, FlatVec};

use super::super::Aligner;

/// A multiple sequence alignment (MSA) builder.
pub struct Columnar<T: Number> {
    /// The Needleman-Wunsch aligner.
    aligner: Aligner<T>,
    /// The columns of the partial MSA.
    columns: Vec<Vec<u8>>,
}

impl<T: Number> Index<usize> for Columnar<T> {
    type Output = Vec<u8>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.columns[index]
    }
}

impl<T: Number> Columnar<T> {
    /// Create a new MSA builder.
    #[must_use]
    pub fn new(aligner: &Aligner<T>) -> Self {
        Self {
            aligner: aligner.clone(),
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
    pub fn with_binary_tree<I, D, C>(self, c: &PermutedBall<T, C>, data: &D) -> Self
    where
        I: AsRef<[u8]>,
        D: Dataset<I>,
        C: Cluster<T>,
    {
        if c.children().is_empty() {
            self.with_cluster(c, data)
        } else {
            if c.children().len() != 2 {
                unreachable!("Binary tree has more than two children.");
            }
            let aligner = self.aligner;
            let left = c.children()[0];
            let right = c.children()[1];

            let l_msa = Self::new(&aligner).with_binary_tree(left, data);
            let r_msa = Self::new(&aligner).with_binary_tree(right, data);

            let l_center = left
                .iter_indices()
                .position(|i| i == left.arg_center())
                .unwrap_or_else(|| unreachable!("Left center not found"));
            let r_center = right
                .iter_indices()
                .position(|i| i == right.arg_center())
                .unwrap_or_else(|| unreachable!("Right center not found"));

            l_msa.merge(l_center, r_msa, r_center)
        }
    }

    /// Add a tree of `Cluster`s to the MSA.
    #[must_use]
    pub fn with_tree<I, D, C>(self, c: &PermutedBall<T, C>, data: &D) -> Self
    where
        I: AsRef<[u8]>,
        D: Dataset<I>,
        C: Cluster<T>,
    {
        if c.children().is_empty() {
            self.with_cluster(c, data)
        } else {
            let children = c.children();
            let (&first, rest) = children.split_first().unwrap_or_else(|| unreachable!("No children"));

            let f_center = first
                .iter_indices()
                .position(|i| i == first.arg_center())
                .unwrap_or_else(|| unreachable!("First center not found"));
            let first = Self::new(&self.aligner).with_tree(first, data);

            let (_, merged) = rest
                .iter()
                .map(|&o| {
                    let o_center = o
                        .iter_indices()
                        .position(|i| i == o.arg_center())
                        .unwrap_or_else(|| unreachable!("Other center not found"));
                    (o_center, Self::new(&self.aligner).with_tree(o, data))
                })
                .fold((f_center, first), |(a_center, acc), (o_center, o)| {
                    (a_center, acc.merge(a_center, o, o_center))
                });

            merged
        }
    }

    /// Replaces all sequences in the MSA with the given sequence.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The sequence to add.
    #[must_use]
    pub fn with_sequence<I: AsRef<[u8]>>(mut self, sequence: &I) -> Self {
        self.columns = sequence.as_ref().iter().map(|&c| vec![c]).collect();
        self
    }

    /// Adds sequences from a `Cluster` to the MSA.
    #[must_use]
    pub fn with_cluster<I, D, C>(self, c: &C, data: &D) -> Self
    where
        I: AsRef<[u8]>,
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
        let first = Self::new(&self.aligner).with_sequence(data.get(first));
        rest.iter()
            .map(|&i| data.get(i))
            .map(|s| Self::new(&self.aligner).with_sequence(s))
            .fold(first, |acc, s| acc.merge(0, s, 0))
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

    /// Get the columns of the MSA.
    #[must_use]
    pub fn columns(&self) -> &[Vec<u8>] {
        &self.columns
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
    pub fn merge(mut self, s_center: usize, mut other: Self, o_center: usize) -> Self {
        ftlog::trace!(
            "Merging MSAs with cardinalities: {} and {}, and centers {s_center} and {o_center}",
            self.len(),
            other.len()
        );
        let s_center = self.get_sequence(s_center);
        let o_center = other.get_sequence(o_center);

        let table = self.aligner.dp_table(&s_center, &o_center);
        let [s_to_o, o_to_s] = self.aligner.alignment_gaps(&s_center, &o_center, &table);

        for i in s_to_o {
            self.add_gap(i).unwrap_or_else(|e| unreachable!("{e}"));
        }

        for i in o_to_s {
            other.add_gap(i).unwrap_or_else(|e| unreachable!("{e}"));
        }

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

    /// Extract the columns as a `FlatVec`.
    #[must_use]
    pub fn to_flat_vec_columns(&self) -> FlatVec<Vec<u8>, usize> {
        FlatVec::new(self.columns.clone())
            .unwrap_or_else(|e| unreachable!("{e}"))
            .with_dim_lower_bound(self.len())
            .with_dim_upper_bound(self.len())
            .with_name("ColWiseMSA")
    }

    /// Extract the rows as a `FlatVec`.
    #[must_use]
    pub fn to_flat_vec_rows(&self) -> FlatVec<Vec<u8>, usize> {
        FlatVec::new(self.extract_msa())
            .unwrap_or_else(|e| unreachable!("{e}"))
            .with_dim_lower_bound(self.width())
            .with_dim_upper_bound(self.width())
            .with_name("RowWiseMSA")
    }
}

impl<T: Number> Columnar<T> {
    /// Parallel version of [`Columnar::with_binary_tree`](crate::msa::dataset::columnar::Columnar::with_binary_tree).
    #[must_use]
    pub fn par_with_binary_tree<I, D, C>(self, c: &PermutedBall<T, C>, data: &D) -> Self
    where
        I: AsRef<[u8]> + Send + Sync,
        D: ParDataset<I>,
        C: ParCluster<T>,
    {
        if c.children().is_empty() {
            self.with_cluster(c, data)
        } else {
            if c.children().len() != 2 {
                unreachable!("Binary tree has more than two children.");
            }
            let aligner = self.aligner;
            let left = c.children()[0];
            let right = c.children()[1];

            let (l_msa, r_msa) = rayon::join(
                || Self::new(&aligner).par_with_binary_tree(left, data),
                || Self::new(&aligner).par_with_binary_tree(right, data),
            );

            let l_center = left
                .iter_indices()
                .position(|i| i == left.arg_center())
                .unwrap_or_else(|| unreachable!("Left center not found"));
            let r_center = right
                .iter_indices()
                .position(|i| i == right.arg_center())
                .unwrap_or_else(|| unreachable!("Right center not found"));

            l_msa.par_merge(l_center, r_msa, r_center)
        }
    }

    /// Parallel version of [`Columnar::with_tree`](crate::msa::dataset::columnar::Columnar::with_tree).
    #[must_use]
    pub fn par_with_tree<I, D, C>(self, c: &PermutedBall<T, C>, data: &D) -> Self
    where
        I: AsRef<[u8]> + Send + Sync,
        D: ParDataset<I>,
        C: ParCluster<T>,
    {
        if c.children().is_empty() {
            self.with_cluster(c, data)
        } else {
            let children = c.children();
            let (&first, rest) = children.split_first().unwrap_or_else(|| unreachable!("No children"));

            let f_center = first
                .iter_indices()
                .position(|i| i == first.arg_center())
                .unwrap_or_else(|| unreachable!("First center not found"));
            let first = Self::new(&self.aligner).with_tree(first, data);

            let (_, merged) = rest
                .par_iter()
                .map(|&o| {
                    let o_center = o
                        .iter_indices()
                        .position(|i| i == o.arg_center())
                        .unwrap_or_else(|| unreachable!("Other center not found"));
                    (o_center, Self::new(&self.aligner).with_tree(o, data))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .fold((f_center, first), |(a_center, acc), (o_center, o)| {
                    (a_center, acc.par_merge(a_center, o, o_center))
                });

            merged
        }
    }

    /// Parallel version of [`Columnar::merge`](crate::msa::dataset::columnar::Columnar::merge).
    #[must_use]
    pub fn par_merge(mut self, s_center: usize, mut other: Self, o_center: usize) -> Self {
        ftlog::trace!(
            "Parallel Merging MSAs with cardinalities: {} and {}, and centers {s_center} and {o_center}",
            self.len(),
            other.len()
        );
        let s_center = self.get_sequence(s_center);
        let o_center = other.get_sequence(o_center);
        let table = self.aligner.dp_table(&s_center, &o_center);
        let [s_to_o, o_to_s] = self.aligner.alignment_gaps(&s_center, &o_center, &table);

        for i in s_to_o {
            self.add_gap(i).unwrap_or_else(|e| unreachable!("{e}"));
        }

        for i in o_to_s {
            other.add_gap(i).unwrap_or_else(|e| unreachable!("{e}"));
        }

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
