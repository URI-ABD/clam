//! Tests for the `Sequence` trait from the `musals` feature.

#![expect(clippy::cast_possible_truncation)]

use abd_clam::{
    common_metrics,
    musals::{AlignedSequence, CostMatrix, Edit, Edits},
};

mod common;

#[test]
fn aligned_sequence() {
    let seq_str = "ACGT";
    let al_seq = AlignedSequence::from(seq_str);
    assert_eq!(al_seq.chunk_count(), 1, "Aligned sequence: {al_seq:?}");

    let gaps = vec![1, 3];
    let gapped_al_seq = al_seq.with_gaps(&gaps);
    assert_eq!(gapped_al_seq.to_string(), "A-CG-T".to_string());
    assert_eq!(gapped_al_seq.chunk_count(), 3, "Gapped sequence: {gapped_al_seq:?}");

    let seq_iter = gapped_al_seq.iter();
    assert_eq!(seq_iter.collect::<String>(), "A-CG-T".to_string());

    let new_seq = AlignedSequence::new_aligned("A-CG-T".chars());
    assert_eq!(gapped_al_seq, new_seq);
}

#[test]
fn edits() {
    let cost_matrix = CostMatrix::<u8>::default();
    let seq = AlignedSequence::from("ACGT");

    // Apply a single substitution edit and check that the resulting sequence is correct.
    let edits = Edits(vec![(1, Edit::Sub('T'))]);
    let mut edited_seq = seq.apply_edits(&edits);
    assert_eq!(edited_seq.to_string(), "ATGT".to_string());

    // Compute the dp table and edits for the edited sequence, and check that applying the edits to one sequence gives the other sequence.
    let [left, right] = seq.nw_edit_scripts(&edited_seq, &cost_matrix);
    assert_eq!(left, Edits(vec![(1, Edit::Sub('T'))]));
    assert_eq!(right, Edits(vec![(1, Edit::Sub('C'))]));
    assert_eq!(edited_seq.to_string(), seq.apply_edits(&left).to_string());
    assert_eq!(seq.to_string(), edited_seq.apply_edits(&right).to_string());

    // Compound the edits by applying an insertion edit to the already edited sequence, and check that the resulting sequence is correct.
    let edits = Edits(vec![(2, Edit::Ins('A'))]);
    edited_seq = edited_seq.apply_edits(&edits);
    assert_eq!(edited_seq.to_string(), "ATAGT".to_string());

    // Verify against the original sequence.
    let [left, right] = seq.nw_edit_scripts(&edited_seq, &cost_matrix);
    // We only check for the lengths of the edits here, since there are multiple valid edit scripts that could transform one sequence into the other.
    assert_eq!(left.len(), 2);
    assert_eq!(right.len(), 2);
    assert_eq!(edited_seq.to_string(), seq.apply_edits(&left).to_string());
    assert_eq!(seq.to_string(), edited_seq.apply_edits(&right).to_string());

    // Compound the edits again by applying a deletion edit to the already edited sequence, and check that the resulting sequence is correct.
    let edits = Edits(vec![(4, Edit::Del)]);
    edited_seq = edited_seq.apply_edits(&edits);
    assert_eq!(edited_seq.to_string(), "ATAG".to_string());

    // Verify against the original sequence again.
    let [left, right] = seq.nw_edit_scripts(&edited_seq, &cost_matrix);
    // We only check for the lengths of the edits here, since there are multiple valid edit scripts that could transform one sequence into the other.
    assert_eq!(left.len(), 3);
    assert_eq!(right.len(), 3);
    assert_eq!(seq.to_string(), edited_seq.apply_edits(&right).to_string());
    assert_eq!(edited_seq.to_string(), seq.apply_edits(&left).to_string());
}

#[test]
fn gaps() {
    let seq = AlignedSequence::from("ACGT");

    let gapped_last = seq.with_gaps(&[4]);
    assert_eq!(gapped_last.to_string(), "ACGT-".to_string());

    let gaps = vec![1, 3];
    let gapped_seq = seq.with_gaps(&gaps);
    assert_eq!(gapped_seq.to_string(), "A-CG-T");

    let recovered_seq = gapped_seq.original();
    assert_eq!(recovered_seq, seq.to_string());

    let seq1 = seq;
    let seq2 = AlignedSequence::from("AGT");
    let cost_matrix = CostMatrix::<u8>::default();

    let [left, right] = seq1.compute_gap_indices(&seq2, &cost_matrix);
    assert!(left.is_empty());
    assert_eq!(right, vec![1]);

    let gapped_seq1 = seq1.with_gaps(&left);
    let gapped_seq2 = seq2.with_gaps(&right);
    assert_eq!(gapped_seq1.len(), gapped_seq2.len());
    assert_eq!(gapped_seq1.to_string(), "ACGT".to_string());
    assert_eq!(gapped_seq2.to_string(), "A-GT".to_string());

    let seq1 = AlignedSequence::from("ACGT");
    let seq2 = AlignedSequence::from("ATAGT");
    let [left, right] = seq1.compute_gap_indices(&seq2, &cost_matrix);
    assert_eq!(left, vec![1]);
    assert!(right.is_empty());
    let gapped_seq1 = seq1.with_gaps(&left);
    let gapped_seq2 = seq2.with_gaps(&right);
    assert_eq!(gapped_seq1.len(), gapped_seq2.len());
    assert_eq!(gapped_seq1.to_string(), "A-CGT".to_string());
    assert_eq!(gapped_seq2.to_string(), "ATAGT".to_string());
}

#[test]
fn distance() {
    let seq1 = "NAJIBEATSPEPPERS";
    let seq2 = "NAJIBPEPPERSEATS";

    let lev_distance = common_metrics::levenshtein_strings(seq1, seq2) as u8;

    let seq1 = AlignedSequence::from(seq1);
    let seq2 = AlignedSequence::from(seq2);
    let cost_matrix = CostMatrix::<u8>::default();
    let nw_distance_12 = seq1.nw_distance(&seq2, &cost_matrix);
    assert_eq!(lev_distance, nw_distance_12);

    let nw_distance_21 = seq2.nw_distance(&seq1, &cost_matrix);
    assert_eq!(lev_distance, nw_distance_21);
}
