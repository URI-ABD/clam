//! Multiple Sequence Alignment At Scale (`MuSAlS`) with CLAM.
//!
//! This module provides functionality to compute multiple sequence alignments (MSA) of genomic and protein sequences using the [`Tree::into_msa`] and
//! [`Tree::par_into_msa`] methods. It also provides the [`Sequence`] trait, which abstracts over the specific sequence types that can be aligned and have edit
//! distances computed among them.
//!
//! Once an MSA is computed, the [`QualityMetric`] and [`QualityMetricResult`] enums provide functionality to compute various quality metrics of the MSA.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::{Dataset, DistanceValue, Tree};

mod alignment;
mod quality_metrics;

pub use alignment::{CostMatrix, Sequence};
pub use quality_metrics::{QualityMetric, QualityMetricResult};

use alignment::PartialMsa;

/// Extension of [`Tree`], gated behind the `musals` feature, providing methods for computing multiple sequence alignments and computing MSA quality metrics.
impl<D, M, T, A> Tree<D, M, T, A>
where
    D: Dataset,
    D::Item: Sequence,
    T: DistanceValue,
{
    /// Aligns all sequences in the tree into a multiple sequence alignment (MSA).
    #[expect(clippy::missing_panics_doc)]
    pub fn into_msa(mut self, cost_matrix: &CostMatrix<T>) -> Self {
        ftlog::info!("Computing MSA for tree with {} sequences.", self.dataset.cardinality());
        // In the following, `ci` is the center index of a cluster, and `pci` is the parent center index of a cluster.

        let (leaves, parents): (Vec<_>, Vec<_>) = self.cluster_map.iter().map(|(&ci, cluster)| (ci, cluster)).partition(|(_, c)| c.is_leaf());

        // Map of parent clusters waiting for their children's alignments to be completed.
        let mut parents_in_waiting = parents
            .into_iter()
            .map(|(ci, parent)| {
                let n_children = parent.child_center_indices().map_or(0, <[usize]>::len);
                (ci, (n_children, Vec::with_capacity(n_children), parent))
            })
            .collect::<HashMap<_, _>>();

        // Initial frontier of leaf clusters with their alignments and indices of their centers and their parent's centers.
        let mut frontier = leaves
            .into_iter()
            .map(|(ci, leaf)| {
                let pci = leaf.parent_center_index();
                let leaf_items = &self.dataset.as_slice()[leaf.items_indices()];
                let msa = PartialMsa::from_leaf(leaf, leaf_items, cost_matrix);
                (ci, pci, msa)
            })
            .collect::<Vec<_>>();

        // When only the root remains, we're done. In the mean time, it is impossible for the frontier to have fewer than one element.
        while !parents_in_waiting.is_empty() {
            assert!(!frontier.is_empty(), "Frontier should not be empty while parents are still in waiting.");

            // Process all clusters in the current frontier and move their alignments to their parents in waiting.
            for (ci, pci, msa) in frontier {
                let pci = pci.unwrap_or(0);

                let (pending_children, child_alignments, _) = parents_in_waiting
                    .get_mut(&pci)
                    .unwrap_or_else(|| unreachable!("Parent cluster with index {pci} not found in parents_in_waiting."));

                // Add the child alignment to the parent in waiting.
                child_alignments.push((ci, msa));
                *pending_children -= 1;
            }

            // Find all parents that have received all their children's alignments.
            let full_parents: HashMap<_, _>;
            (full_parents, parents_in_waiting) = parents_in_waiting.into_iter().partition(|(_, (pending_children, _, _))| *pending_children == 0);

            frontier = full_parents
                .into_iter()
                .map(|(_, (_, mut child_alignments, parent))| {
                    // Sort child alignments by center index to ensure consistent ordering.
                    child_alignments.sort_by_key(|(idx, _)| *idx);

                    // Align the parent cluster.
                    let child_alignments = child_alignments.into_iter().map(|(_, msa)| msa).collect::<Vec<_>>();
                    let parent_center = &self.dataset.as_slice()[parent.center_index()].1;
                    let parent_msa = PartialMsa::from_parent(parent_center, child_alignments, cost_matrix);

                    // Return the aligned parent MSA with its center and parent center indices.
                    let ci = parent.center_index();
                    let pci = parent.parent_center_index();
                    (ci, pci, parent_msa)
                })
                .collect();
        }

        // Final sanity checks and extraction of the MSA.
        assert_eq!(frontier.len(), 1, "The root cluster should now be in the frontier.");

        // Get the final MSA from the frontier and add the center of the root cluster.
        let (ci, pci, msa) = frontier.pop().unwrap_or_else(|| unreachable!("Frontier should contain the root cluster."));
        assert_eq!(ci, 0, "The root cluster should have center index 0.");
        assert!(pci.is_none(), "The root cluster should have no parent center index.");
        assert_eq!(
            self.dataset.cardinality(),
            msa.n_seq(),
            "Number of sequences in the final MSA should match the original."
        );

        let aligned_rows = msa.into_rows(true);
        self.dataset.as_mut_slice().iter_mut().zip(aligned_rows).for_each(|((_, prev_seq), new_seq)| {
            *prev_seq = new_seq;
        });

        self
    }
}

/// Parallel versions of the MSA methods for the `musals` feature.
impl<D, M, T, A> Tree<D, M, T, A>
where
    D: Dataset + Send + Sync,
    D::Id: Send + Sync,
    D::Item: Sequence + Send + Sync,
    M: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
{
    /// Parallel version of [`Self::into_msa`].
    #[expect(clippy::missing_panics_doc)]
    pub fn par_into_msa(mut self, cost_matrix: &CostMatrix<T>) -> Self {
        ftlog::info!("Computing MSA for tree with {} sequences in parallel.", self.dataset.cardinality());
        // In the following, `ci` is the center index of a cluster, and `pci` is the parent center index of a cluster.

        let (leaves, parents): (Vec<_>, Vec<_>) = self.cluster_map.par_iter().map(|(&ci, cluster)| (ci, cluster)).partition(|(_, c)| c.is_leaf());

        // Map of parent clusters waiting for their children's alignments to be completed.
        let mut parents_in_waiting = parents
            .into_par_iter()
            .map(|(ci, parent)| {
                let n_children = parent.child_center_indices().map_or(0, <[usize]>::len);
                (ci, (n_children, Vec::with_capacity(n_children), parent))
            })
            .collect::<HashMap<_, _>>();

        // Initial frontier of leaf clusters with their alignments and indices of their centers and their parent's centers.
        let mut frontier = leaves
            .into_par_iter()
            .map(|(ci, leaf)| {
                let pci = leaf.parent_center_index();
                let leaf_items = &self.dataset.as_slice()[leaf.items_indices()];
                let msa = PartialMsa::par_from_leaf(leaf, leaf_items, cost_matrix);
                (ci, pci, msa)
            })
            .collect::<Vec<_>>();

        // When only the root remains, we're done. In the mean time, it is impossible for the frontier to have fewer than one element.
        while !parents_in_waiting.is_empty() {
            assert!(!frontier.is_empty(), "Frontier should not be empty while parents are still in waiting.");

            // Process all clusters in the current frontier and move their alignments to their parents in waiting.
            for (ci, pci, msa) in frontier {
                let pci = pci.unwrap_or(0);

                let (pending_children, child_alignments, _) = parents_in_waiting
                    .get_mut(&pci)
                    .unwrap_or_else(|| unreachable!("Parent cluster with index {pci} not found in parents_in_waiting."));

                // Add the child alignment to the parent in waiting.
                child_alignments.push((ci, msa));
                *pending_children -= 1;
            }

            // Find all parents that have received all their children's alignments.
            let full_parents: Vec<_>; // `rayon`'s `partition` allows collecting into different types.
            (full_parents, parents_in_waiting) = parents_in_waiting
                .into_par_iter()
                .partition(|(_, (pending_children, _, _))| *pending_children == 0);

            frontier = full_parents
                .into_par_iter()
                .map(|(_, (_, mut child_alignments, parent))| {
                    // Sort child alignments by center index to ensure consistent ordering.
                    child_alignments.sort_by_key(|(idx, _)| *idx);

                    // Align the parent cluster.
                    let child_alignments = child_alignments.into_iter().map(|(_, msa)| msa).collect::<Vec<_>>();
                    let parent_center = &self.dataset.as_slice()[parent.center_index()].1;
                    let parent_msa = PartialMsa::par_from_parent(parent_center, child_alignments, cost_matrix);

                    // Return the aligned parent MSA with its center and parent center indices.
                    let ci = parent.center_index();
                    let pci = parent.parent_center_index();
                    (ci, pci, parent_msa)
                })
                .collect();
        }

        // Final sanity checks and extraction of the MSA.
        assert_eq!(frontier.len(), 1, "The root cluster should now be in the frontier.");

        // Get the final MSA from the frontier and add the center of the root cluster.
        let (ci, pci, msa) = frontier.pop().unwrap_or_else(|| unreachable!("Frontier should contain the root cluster."));
        assert_eq!(ci, 0, "The root cluster should have center index 0.");
        assert!(pci.is_none(), "The root cluster should have no parent center index.");
        assert_eq!(
            self.dataset.cardinality(),
            msa.n_seq(),
            "Number of aligned sequences should match the original."
        );

        let aligned_rows = msa.par_into_rows(true);
        self.dataset
            .as_mut_slice()
            .par_iter_mut()
            .zip(aligned_rows)
            .for_each(|((_, prev_seq), new_seq)| {
                *prev_seq = new_seq;
            });

        self
    }
}

#[cfg(test)]
mod tests {
    use distances::strings::levenshtein;
    use rand::prelude::*;
    use test_case::test_case;

    use crate::{Dataset, Tree};

    use super::{CostMatrix, Sequence};

    fn check_sequences_equal<D, M, T, A>(original: &Tree<D, M, T, A>, aligned: &Tree<D, M, T, A>, mode: &str)
    where
        D: Dataset + Send + Sync,
        D::Item: Sequence + Eq + core::fmt::Debug,
        D::Id: Eq + core::fmt::Debug,
    {
        assert_eq!(
            original.dataset.cardinality(),
            aligned.dataset.cardinality(),
            "Number of sequences should match in {} mode.",
            mode
        );

        let max_len = original.dataset.as_slice().iter().map(|(_, seq)| seq.as_ref().len()).max().unwrap_or(0);
        let aligned_max_len = aligned.dataset.as_slice().iter().map(|(_, seq)| seq.as_ref().len()).max().unwrap_or(0);
        assert!(
            aligned_max_len >= max_len,
            "Aligned sequences should be at least as long as the longest original sequence in {} mode.",
            mode
        );
        assert!(
            aligned_max_len <= max_len * 2,
            "Aligned sequences should be at most twice as long as the longest original sequence in {} mode.",
            mode
        );

        let o_sequences = original.dataset.as_slice().iter().map(|(_, seq)| seq.clone()).collect::<Vec<_>>();
        let a_sequences = aligned.dataset.as_slice().iter().map(|(_, seq)| seq.without_gaps()).collect::<Vec<_>>();

        assert_eq!(o_sequences.len(), a_sequences.len(), "Number of sequences should match in {} mode.", mode);
        assert_eq!(o_sequences, a_sequences, "Sequences should match after alignment in {} mode.", mode);

        for (i, ((o_id, o_seq), (a_id, a_seq))) in original.dataset.as_slice().iter().zip(aligned.dataset.as_slice().iter()).enumerate() {
            assert_eq!(o_id, a_id, "Sequence IDs at index {} do not match after alignment in {} mode.", i, mode);
            assert_eq!(
                o_seq,
                &a_seq.without_gaps(),
                "Sequence at index {} does not match after removing gaps in {} mode.",
                i,
                mode
            );
        }
    }

    #[test]
    fn test_msa_small() -> Result<(), String> {
        let metric = |a: &String, b: &String| levenshtein::<u8>(a, b);
        let cost_matrix = CostMatrix::<u8>::default();
        let sequences = vec![
            "ACTGA".to_string(),
            "CTGAA".to_string(),
            "TGAAC".to_string(),
            "GAACT".to_string(),
            "AACTG".to_string(),
        ];
        let sequences = sequences.into_iter().enumerate().collect::<Vec<_>>();

        let tree = Tree::new_minimal(sequences.clone(), &metric)?;
        let msa_tree = tree.clone().into_msa(&cost_matrix);
        check_sequences_equal(&tree, &msa_tree, "recursive");

        let par_tree = Tree::par_new_minimal(sequences, metric)?;
        let par_msa_tree = par_tree.clone().par_into_msa(&cost_matrix);
        check_sequences_equal(&par_tree, &par_msa_tree, "parallel recursive");

        Ok(())
    }

    #[test_case(20)]
    #[test_case(50)]
    #[test_case(100)]
    fn test_msa_medium(car: usize) -> Result<(), String> {
        let metric = |a: &String, b: &String| levenshtein::<u16>(a, b);
        let cost_matrix = CostMatrix::<u16>::default();

        let (min_len, max_len) = (8, 12);
        let characters = ['A', 'C', 'G', 'T'];
        let mut rng = rand::rng();
        let sequences = (0..car)
            .map(|_| {
                let len: usize = rng.random_range(min_len..=max_len);
                (0..len).map(|_| characters[rng.random_range(0..characters.len())]).collect::<String>()
            })
            .collect::<Vec<String>>();
        let sequences = sequences.into_iter().enumerate().collect::<Vec<_>>();

        let tree = Tree::new_minimal(sequences.clone(), &metric)?;
        let msa_tree = tree.clone().into_msa(&cost_matrix);
        check_sequences_equal(&tree, &msa_tree, "recursive");

        let par_tree = Tree::par_new_minimal(sequences, metric)?;
        let par_msa_tree = par_tree.clone().par_into_msa(&cost_matrix);
        check_sequences_equal(&par_tree, &par_msa_tree, "parallel recursive");

        Ok(())
    }
}
