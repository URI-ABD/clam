//! An adapter for `Cluster` that stores information about partial alignments.

use core::fmt::Debug;

use distances::number::UInt;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

mod impl_adapter;
mod impl_cluster;

/// A trait for types that can be aligned in a multiple sequence alignment.
///
/// We provide an implementation for `String`.
pub trait Alignable: Sized + Clone + Debug {
    /// Returns the width of the Aligned type in the MSA.
    fn width(&self) -> usize;

    /// Inserts a gap at the given index.
    fn insert_gap(&mut self, idx: usize);

    /// Appends a gap to the end of the alignable type.
    fn append_gap(&mut self);

    /// Applies gaps to the alignable type. The gaps will be added in reverse
    /// order.
    #[must_use]
    fn apply_gaps(mut self, gaps: &[usize]) -> Self {
        for &idx in gaps.iter().rev() {
            if idx == self.width() {
                self.append_gap();
            } else {
                self.insert_gap(idx);
            }
        }
        self
    }
}

impl Alignable for String {
    fn width(&self) -> usize {
        self.len()
    }

    fn insert_gap(&mut self, idx: usize) {
        self.insert(idx, '-');
    }

    fn append_gap(&mut self) {
        self.push('-');
    }
}

/// A variant of `Cluster` used to recursively build partial MSAs from the leaves
/// up to the root. At the root level, this will contain the full MSA.
#[derive(Serialize, Deserialize)]
pub struct PartialMSA<I, U: UInt, D: Dataset<I, U>, S: Cluster<I, U, D>> {
    /// The source cluster used to create this partial MSA.
    source: S,
    /// The children of this cluster.
    children: Vec<(usize, U, Box<Self>)>,
    /// The indices at which gaps were added in the left and right children to
    /// bring them into alignment.
    gaps: Gaps,
    /// Phantom data for the compiler.
    _phantom: core::marker::PhantomData<(I, D)>,
}

/// The indices at which gaps were added in the left and right children to bring
/// them into alignment.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Gaps {
    /// The indices at which gaps were added in the left child center.
    l: Vec<usize>,
    /// The indices at which gaps were added in the right child center.
    r: Vec<usize>,
}

impl<I: Alignable, U: UInt, D: Dataset<I, U>, S: Cluster<I, U, D>> PartialMSA<I, U, D, S> {
    /// Returns the indexed point as if it were in a multiple sequence alignment.
    pub fn aligned_point(&self, data: &D, idx: usize) -> I {
        if self.is_leaf() {
            data.get(idx).clone()
        } else {
            let l_child = self.children[0].2.as_ref();
            let r_child = self.children[1].2.as_ref();

            if l_child.contains(idx) {
                l_child.aligned_point(data, idx).apply_gaps(&self.gaps.l)
            } else {
                r_child.aligned_point(data, idx).apply_gaps(&self.gaps.r)
            }
        }
    }

    /// Returns all points in the cluster as if they were in a multiple sequence
    /// alignment.
    ///
    /// The sequences will be returned in the order of a depth-first traversal
    /// of the tree.
    pub fn full_msa(&self, data: &D) -> Vec<I> {
        if self.children.is_empty() {
            self.indices().map(|i| data.get(i).clone()).collect()
        } else {
            self.children()
                .iter()
                .zip([&self.gaps.l, &self.gaps.r])
                .flat_map(|((_, _, child), gaps)| {
                    child
                        .full_msa(data)
                        .into_iter()
                        .map(move |point| point.apply_gaps(gaps))
                })
                .collect()
        }
    }
}

impl<I: Alignable + Send + Sync, U: UInt, D: ParDataset<I, U>, S: ParCluster<I, U, D>> PartialMSA<I, U, D, S> {
    /// Parallel version of `full_msa`.
    pub fn par_full_msa(&self, data: &D) -> Vec<I> {
        if self.children.is_empty() {
            self.indices().map(|i| data.get(i).clone()).collect()
        } else {
            self.children()
                .par_iter()
                .zip([&self.gaps.l, &self.gaps.r])
                .flat_map(|((_, _, child), gaps)| {
                    child
                        .par_full_msa(data)
                        .into_par_iter()
                        .map(move |point| point.apply_gaps(gaps))
                })
                .collect()
        }
    }
}
