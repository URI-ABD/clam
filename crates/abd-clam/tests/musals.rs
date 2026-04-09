//! Tests for multiple sequence alignment.

#![expect(clippy::cast_possible_truncation)]

use abd_clam::{
    DistanceValue, Tree, common_metrics,
    musals::{AlignedSequence, CostMatrix, MusalsTree},
};
use rand::prelude::*;
use test_case::test_case;

mod common;

/// Checks that the two trees have the same sequences (ignoring gaps).
fn check_sequences_equal<Id, T, A, M1, M2>(original: &Tree<Id, String, T, A, M1>, aligned: &MusalsTree<Id, T, A, M2>, mode: &str)
where
    Id: Eq + core::fmt::Debug,
    T: DistanceValue,
{
    assert_eq!(
        original.cardinality(),
        aligned.cardinality(),
        "Number of sequences should match in {mode} mode."
    );

    let max_len = original.iter_items().map(|(_, seq, _)| seq.len()).max().unwrap_or(0);
    let aligned_max_len = aligned.iter_items().map(|(_, seq, _)| seq.original().len()).max().unwrap_or(0);
    assert!(
        aligned_max_len >= max_len,
        "Aligned sequences should be at least as long as the longest original sequence in {mode} mode."
    );
    assert!(
        aligned_max_len <= max_len * 2,
        "Aligned sequences should be at most twice as long as the longest original sequence in {mode} mode."
    );

    let o_sequences = original.iter_items().map(|(_, seq, _)| seq).collect::<Vec<_>>();
    let a_sequences = aligned.iter_items().map(|(_, seq, _)| seq.original()).collect::<Vec<_>>();
    let a_sequences = a_sequences.iter().collect::<Vec<_>>();

    assert_eq!(o_sequences.len(), a_sequences.len(), "Number of sequences should match in {mode} mode.");
    assert_eq!(o_sequences, a_sequences, "Sequences should match after alignment in {mode} mode.");

    for (i, ((o_id, o_seq, _), (a_id, a_seq, _))) in original.iter_items().zip(aligned.iter_items()).enumerate() {
        assert_eq!(o_id, a_id, "Sequence IDs at index {i} do not match after alignment in {mode} mode.");
        assert_eq!(
            o_seq,
            &a_seq.original(),
            "Sequence at index {i} does not match after removing gaps in {mode} mode."
        );
    }
}

#[test]
fn msa_small() -> Result<(), String> {
    let metric_unaligned = |a: &String, b: &String| common_metrics::levenshtein_strings(a, b) as u8;
    let metric_aligned = |a: &AlignedSequence, b: &AlignedSequence| common_metrics::levenshtein_aligned(a, b) as u8;
    let cost_matrix = CostMatrix::<u8>::default();
    let sequences = vec!["ACTGA", "CTGAA", "TGAAC", "GAACT", "AACTG"];

    let sequences = sequences.into_iter().map(String::from).collect::<Vec<_>>();
    let tree = Tree::new_minimal(sequences.clone(), &metric_unaligned)?;
    let msa_tree = tree.clone().align(cost_matrix.clone(), &metric_aligned);
    check_sequences_equal(&tree, &msa_tree, "serial");

    let par_tree = Tree::par_new_minimal(sequences, &metric_unaligned)?;
    let par_msa_tree = par_tree.clone().par_align(cost_matrix, &metric_aligned);
    check_sequences_equal(&par_tree, &par_msa_tree, "parallel");

    Ok(())
}

#[test_case(20)]
#[test_case(50)]
#[test_case(100)]
fn msa_medium(car: usize) -> Result<(), String> {
    let metric_unaligned = |a: &String, b: &String| common_metrics::levenshtein_strings(a, b) as u16;
    let metric_aligned = |a: &AlignedSequence, b: &AlignedSequence| common_metrics::levenshtein_aligned(a, b) as u16;
    let cost_matrix = CostMatrix::<u16>::default();

    let (min_len, max_len) = (8, 12);
    let characters = ['A', 'C', 'G', 'T'];
    let mut rng = rand::rng();
    let sequences = (0..car)
        .map(|_| {
            let len: usize = rng.random_range(min_len..=max_len);
            (0..len).map(|_| characters[rng.random_range(0..characters.len())]).collect::<String>()
        })
        .collect::<Vec<_>>();

    let tree = Tree::new_minimal(sequences.clone(), &metric_unaligned)?;
    let msa_tree = tree.clone().align(cost_matrix.clone(), &metric_aligned);
    check_sequences_equal(&tree, &msa_tree, "serial");

    let par_tree = Tree::par_new_minimal(sequences, &metric_unaligned)?;
    let par_msa_tree = par_tree.clone().par_align(cost_matrix, &metric_aligned);
    check_sequences_equal(&par_tree, &par_msa_tree, "parallel");

    Ok(())
}
