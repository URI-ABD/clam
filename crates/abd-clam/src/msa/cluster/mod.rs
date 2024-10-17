//! An adapter for `Cluster` that stores information about partial alignments.

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

mod impl_adapter;
mod impl_cluster;

/// A trait for types that can be aligned in a multiple sequence alignment.
///
/// We provide an implementation for `String`.
pub trait Alignable: Sized + Clone {
    /// Inserts a gap at the given index.
    fn insert_gap(&mut self, idx: usize);

    /// Applies gaps to the alignable type. The gaps will be added in reverse
    /// order.
    #[must_use]
    fn apply_gaps(mut self, gap_ids: &[usize]) -> Self {
        for &idx in gap_ids.iter().rev() {
            self.insert_gap(idx);
        }
        self
    }
}

impl Alignable for String {
    fn insert_gap(&mut self, idx: usize) {
        self.insert(idx, '-');
    }
}

/// A variant of `Cluster` used to recursively build partial MSAs from the leaves
/// up to the root. At the root level, this will contain the full MSA.
pub struct PartialMSA<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> {
    /// The source cluster used to create this partial MSA.
    source: S,
    /// The children of this cluster.
    children: Vec<(usize, U, Box<Self>)>,
    /// The indices at which gaps were added in the left and right children to
    /// bring them into alignment.
    params: GapIds,
    /// Phantom data for the compiler.
    _phantom: core::marker::PhantomData<(I, D)>,
}

/// The indices at which gaps were added in the left and right children to bring
/// them into alignment.
#[derive(Debug, Default)]
struct GapIds {
    /// The indices at which gaps were added in the left child center.
    left: Vec<usize>,
    /// The indices at which gaps were added in the right child center.
    right: Vec<usize>,
}

impl<I: Alignable, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> PartialMSA<I, U, D, S> {
    /// Returns the indexed point as if it were in a multiple sequence alignment.
    pub fn aligned_point(&self, data: &D, index: usize) -> I {
        if self.children.is_empty() {
            data.get(index).clone()
        } else {
            let l_child = self.children[0].2.as_ref();
            let r_child = self.children[1].2.as_ref();

            let (gap_ids, aligned) = if l_child.indices().any(|i| i == index) {
                (&self.params.left, l_child.aligned_point(data, index))
            } else {
                (&self.params.right, r_child.aligned_point(data, index))
            };

            aligned.apply_gaps(gap_ids)
        }
    }

    /// Returns all points in the cluster as if they were in a multiple sequence
    /// alignment.
    pub fn full_msa(&self, data: &D) -> Vec<I> {
        if self.children.is_empty() {
            self.indices().map(|i| data.get(i).clone()).collect()
        } else {
            let l_msa = self.children[0].2.full_msa(data);
            let l_gap_ids = &self.params.left;

            let r_msa = self.children[1].2.full_msa(data);
            let r_gap_ids = &self.params.right;

            l_msa
                .into_iter()
                .map(|point| point.apply_gaps(l_gap_ids))
                .chain(r_msa.into_iter().map(|point| point.apply_gaps(r_gap_ids)))
                .collect()
        }
    }
}

impl<I: Alignable + Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<I, U, D>> PartialMSA<I, U, D, S> {
    /// Parallel version of `full_msa`.
    pub fn par_full_msa(&self, data: &D) -> Vec<I> {
        if self.children.is_empty() {
            self.indices().map(|i| data.get(i).clone()).collect()
        } else {
            let l_msa = self.children[0].2.full_msa(data);
            let l_gap_ids = &self.params.left;

            let r_msa = self.children[1].2.full_msa(data);
            let r_gap_ids = &self.params.right;

            l_msa
                .into_par_iter()
                .map(|point| point.apply_gaps(l_gap_ids))
                .chain(r_msa.into_par_iter().map(|point| point.apply_gaps(r_gap_ids)))
                .collect()
        }
    }
}
