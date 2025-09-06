#![allow(missing_docs)]

use distances::strings::{Edit, aligned_x_to_y, aligned_x_to_y_no_sub, apply_edits, unaligned_x_to_y, x_to_y_alignment, x2y_helper};

#[test]
fn tiny_aligned() {
    let x = "A-C";
    let y = "AAC";

    let actual = x2y_helper(x, y);
    let expected = vec![Edit::Sub(1, 'A')];

    assert_eq!(actual, expected);
}

#[test]
fn medium_aligned() {
    let x = "NAJIB-PEPPERSEATS";
    let y = "NAJIBEATSPEPPE-RS";

    let actual = x2y_helper(x, y);
    let expected = vec![
        Edit::Sub(5, 'E'),
        Edit::Sub(6, 'A'),
        Edit::Sub(7, 'T'),
        Edit::Sub(8, 'S'),
        Edit::Sub(11, 'P'),
        Edit::Sub(12, 'P'),
        Edit::Sub(14, '-'),
        Edit::Sub(15, 'R'),
    ];

    assert_eq!(actual, expected);
}

#[test]
fn tiny_unaligned() {
    let x = "A-C";
    let y = "AAC";

    let actual = unaligned_x_to_y(x, y);
    let expected = vec![Edit::Ins(1, 'A')];

    assert_eq!(actual, expected);
}

#[test]
fn small_unaligned() {
    let x = "A-CBAAB";
    let y = "AACA-AC";

    let actual = unaligned_x_to_y(x, y);
    let expected = vec![Edit::Ins(1, 'A'), Edit::Sub(3, 'A'), Edit::Del(4), Edit::Sub(5, 'C')];

    assert_eq!(actual, expected);
}

#[test]
fn test_apply_edits() {
    let x = "ACBAAB";
    let edits: Vec<Edit> = vec![Edit::Ins(1, 'A'), Edit::Sub(3, 'A'), Edit::Del(4), Edit::Sub(5, 'C')];

    let actual = apply_edits(x, &edits);

    let expected = "AACAAC";

    assert_eq!(actual, expected);
}

#[test]
fn test_alignment() {
    let seq_1_10 = "CAGAATATTA";
    let seq_2_10 = "TTGCTTTGAT";
    let seq_1_20 = "GAAAGCCTATCGTCTGAGCG";
    let seq_2_20 = "AAGGGACGCGTTGGAGTTAC";
    let edits_10_10 = aligned_x_to_y(seq_1_10, seq_2_10);

    let mut deletes: Vec<usize> = vec![3];
    let mut inserts: Vec<(usize, char)> = vec![(9, 'T')];
    let mut substitutions: Vec<(usize, char)> = vec![(0, 'T'), (1, 'T'), (3, 'C'), (5, 'T'), (7, 'G')];
    assert_eq!(edits_10_10.len(), 7);
    for edit in edits_10_10 {
        match edit {
            Edit::Del(x) => {
                assert_eq!(deletes[0], x);
                deletes.remove(0);
            }
            Edit::Ins(x, c) => {
                assert!(inserts[0].0 == x && inserts[0].1 == c);
                inserts.remove(0);
            }
            Edit::Sub(x, c) => {
                assert!(substitutions[0].0 == x && substitutions[0].1 == c);
                substitutions.remove(0);
            }
        }
    }
    assert_eq!(deletes.len(), 0);
    assert_eq!(inserts.len(), 0);
    assert_eq!(substitutions.len(), 0);

    let edits_10_20 = aligned_x_to_y(seq_1_10, seq_2_20);
    let mut deletes: Vec<usize> = vec![];
    let mut inserts: Vec<(usize, char)> = vec![
        (2, 'G'),
        (3, 'G'),
        (6, 'C'),
        (7, 'G'),
        (8, 'C'),
        (9, 'G'),
        (12, 'G'),
        (13, 'G'),
        (15, 'G'),
        (19, 'C'),
    ];
    let mut substitutions: Vec<(usize, char)> = vec![(0, 'A'), (10, 'T')];
    assert_eq!(edits_10_20.len(), 12);
    for edit in edits_10_20 {
        match edit {
            Edit::Del(x) => {
                assert_eq!(deletes[0], x);
                deletes.remove(0);
            }
            Edit::Ins(x, c) => {
                assert!(inserts[0].0 == x && inserts[0].1 == c);
                inserts.remove(0);
            }
            Edit::Sub(x, c) => {
                assert!(substitutions[0].0 == x && substitutions[0].1 == c);
                substitutions.remove(0);
            }
        }
    }
    assert_eq!(deletes.len(), 0);
    assert_eq!(inserts.len(), 0);
    assert_eq!(substitutions.len(), 0);

    let mut deletes: Vec<usize> = vec![0, 0, 3, 4, 4, 5, 5, 6, 9, 9];
    let mut substitutions: Vec<(usize, char)> = vec![(0, 'T'), (1, 'T'), (9, 'T')];
    let edits_20_10 = aligned_x_to_y(seq_1_20, seq_2_10);
    assert_eq!(edits_20_10.len(), 13);
    for edit in edits_20_10 {
        match edit {
            Edit::Del(x) => {
                assert_eq!(deletes[0], x);
                deletes.remove(0);
            }
            Edit::Ins(x, c) => {
                assert!(inserts[0].0 == x && inserts[0].1 == c);
                inserts.remove(0);
            }
            Edit::Sub(x, c) => {
                assert!(substitutions[0].0 == x && substitutions[0].1 == c);
                substitutions.remove(0);
            }
        }
    }
    assert_eq!(deletes.len(), 0);
    assert_eq!(inserts.len(), 0);
    assert_eq!(substitutions.len(), 0);

    let edits_20_20 = aligned_x_to_y(seq_1_20, seq_2_20);
    let mut deletes: Vec<usize> = vec![0, 0];
    let mut inserts: Vec<(usize, char)> = vec![(16, 'T'), (17, 'T')];
    let mut substitutions: Vec<(usize, char)> = vec![(3, 'G'), (4, 'G'), (5, 'A'), (6, 'C'), (7, 'G'), (11, 'T'), (12, 'G'), (18, 'A'), (19, 'C')];
    assert_eq!(edits_20_20.len(), 13);
    for edit in edits_20_20 {
        match edit {
            Edit::Del(x) => {
                assert_eq!(deletes[0], x);
                deletes.remove(0);
            }
            Edit::Ins(x, c) => {
                assert!(inserts[0].0 == x && inserts[0].1 == c);
                inserts.remove(0);
            }
            Edit::Sub(x, c) => {
                assert!(substitutions[0].0 == x && substitutions[0].1 == c);
                substitutions.remove(0);
            }
        }
    }
    assert_eq!(deletes.len(), 0);
    assert_eq!(inserts.len(), 0);
    assert_eq!(substitutions.len(), 0);
}

#[test]
fn test_alignment_no_sub() {
    let seq_1_10 = "CAGAATATTA";
    let seq_2_10 = "TTGCTTTGAT";
    let seq_1_20 = "GAAAGCCTATCGTCTGAGCG";
    let seq_2_20 = "AAGGGACGCGTTGGAGTTAC";
    let edits_10_10 = aligned_x_to_y_no_sub(seq_1_10, seq_2_10);

    let mut deletes: Vec<usize> = vec![3];
    let mut inserts: Vec<(usize, char)> = vec![(9, 'T')];
    assert_eq!(edits_10_10.len(), 2);
    for edit in edits_10_10 {
        match edit {
            Edit::Del(x) => {
                assert_eq!(deletes[0], x);
                deletes.remove(0);
            }
            Edit::Ins(x, c) => {
                assert!(inserts[0].0 == x && inserts[0].1 == c);
                inserts.remove(0);
            }
            _ => assert_eq!(0, 1),
        }
    }
    assert_eq!(deletes.len(), 0);
    assert_eq!(inserts.len(), 0);

    let edits_10_20 = aligned_x_to_y_no_sub(seq_1_10, seq_2_20);
    let mut deletes: Vec<usize> = vec![];
    let mut inserts: Vec<(usize, char)> = vec![
        (2, 'G'),
        (3, 'G'),
        (6, 'C'),
        (7, 'G'),
        (8, 'C'),
        (9, 'G'),
        (12, 'G'),
        (13, 'G'),
        (15, 'G'),
        (19, 'C'),
    ];
    assert_eq!(edits_10_20.len(), 10);
    for edit in edits_10_20 {
        match edit {
            Edit::Del(x) => {
                assert_eq!(deletes[0], x);
                deletes.remove(0);
            }
            Edit::Ins(x, c) => {
                assert!(inserts[0].0 == x && inserts[0].1 == c);
                inserts.remove(0);
            }
            _ => assert_eq!(0, 1),
        }
    }

    let mut deletes: Vec<usize> = vec![0, 0, 3, 4, 4, 5, 5, 6, 9, 9];
    let edits_20_10 = aligned_x_to_y_no_sub(seq_1_20, seq_2_10);
    assert_eq!(edits_20_10.len(), 10);
    for edit in edits_20_10 {
        match edit {
            Edit::Del(x) => {
                assert_eq!(deletes[0], x);
                deletes.remove(0);
            }
            _ => assert_eq!(0, 1),
        }
    }
    let edits_20_20 = aligned_x_to_y_no_sub(seq_1_20, seq_2_20);
    let mut deletes: Vec<usize> = vec![0, 0];
    let mut inserts: Vec<(usize, char)> = vec![(16, 'T'), (17, 'T')];
    assert_eq!(edits_20_20.len(), 4);
    for edit in edits_20_20 {
        match edit {
            Edit::Del(x) => {
                assert_eq!(deletes[0], x);
                deletes.remove(0);
            }
            Edit::Ins(x, c) => {
                assert!(inserts[0].0 == x && inserts[0].1 == c);
                inserts.remove(0);
            }
            _ => assert_eq!(0, 1),
        }
    }
}

#[test]
fn test_alignment_gaps() {
    let seq_1_10 = "CAGAATATTA";
    let seq_2_10 = "TTGCTTTGAT";
    let seq_1_20 = "GAAAGCCTATCGTCTGAGCG";
    let seq_2_20 = "AAGGGACGCGTTGGAGTTAC";
    let align_10_10 = x_to_y_alignment(seq_1_10, seq_2_10);
    let gaps_10_10: [Vec<usize>; 2] = [vec![10], vec![3]];
    for (i, g) in align_10_10[0].iter().enumerate() {
        assert_eq!(&gaps_10_10[0][i], g);
    }
    for (i, g) in align_10_10[1].iter().enumerate() {
        assert_eq!(&gaps_10_10[1][i], g);
    }

    let align_10_20 = x_to_y_alignment(seq_1_10, seq_2_20);
    let gaps_10_20: [Vec<usize>; 2] = [vec![2, 3, 6, 7, 8, 9, 12, 13, 15, 19], vec![]];
    assert_eq!(align_10_20[1].len(), 0);
    for (i, g) in align_10_20[0].iter().enumerate() {
        assert_eq!(&gaps_10_20[0][i], g);
    }

    let align_20_10 = x_to_y_alignment(seq_1_20, seq_2_10);
    let gaps_20_10: [Vec<usize>; 2] = [vec![], vec![0, 0, 3, 4, 4, 5, 5, 6, 9, 9]];
    for (i, g) in align_20_10[1].iter().enumerate() {
        assert_eq!(&gaps_20_10[1][i], g);
    }
    assert_eq!(align_20_10[0].len(), 0);

    let align_20_20 = x_to_y_alignment(seq_1_20, seq_2_20);
    let gaps_20_20: [Vec<usize>; 2] = [vec![18, 19], vec![0, 0]];
    for (i, g) in align_20_20[0].iter().enumerate() {
        assert_eq!(&gaps_20_20[0][i], g);
    }
    for (i, g) in align_20_20[1].iter().enumerate() {
        assert_eq!(&gaps_20_20[1][i], g);
    }
}
