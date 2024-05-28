use distances::strings::Edit;
use distances::strings::_x_to_y;
use distances::strings::needleman_wunsch::apply_edits;
use distances::strings::needleman_wunsch::compute_table;
use distances::strings::needleman_wunsch::trace_back_recursive;
use distances::strings::unaligned_x_to_y;
use distances::strings::Penalties;

/// Applies a random edit to a given string.
/// The character (if applicable) for the edit is a random character from the given alphabet.
///
/// # Arguments
///
/// * `string`: The string to apply the edit to.
/// * `alphabet`: The alphabet to choose the character from.
///
/// # Returns
///
/// A string with a random edit applied.
fn apply_random_edit(string: &str, alphabet: &Vec<char>) -> String {
    let edit_type = rand::random::<u8>() % 3;
    let length = string.len();
    let char = alphabet[rand::random::<usize>() % alphabet.len()];

    match edit_type {
        0 => {
            let index = rand::random::<usize>() % (length + 1);
            apply_edits(string, &[Edit::Ins(index, char)])
        }
        1 => apply_edits(string, &[Edit::Del(rand::random::<usize>() % length)]),
        2 => apply_edits(string, &[Edit::Sub(rand::random::<usize>() % length, char)]),
        _ => unreachable!(),
    }
}

#[test]
fn tiny_aligned() {
    let x = "A-C";
    let y = "AAC";

    let actual = _x_to_y(x, y);
    let expected = vec![Edit::Sub(1, 'A')];

    assert_eq!(actual, expected);
}

#[test]
fn medium_aligned() {
    let x = "NAJIB-PEPPERSEATS";
    let y = "NAJIBEATSPEPPE-RS";

    let actual = _x_to_y(x, y);
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
    let expected = vec![
        Edit::Ins(1, 'A'),
        Edit::Sub(3, 'A'),
        Edit::Del(4),
        Edit::Sub(5, 'C'),
    ];

    assert_eq!(actual, expected);
}

#[test]
fn random_reference() {
    let alphabet = vec!['A', 'C', 'G', 'T'];
    let x = "ACCCGAGTCGTTT";

    for _ in 0..50 {
        let mut y = x.to_string();

        for _ in 0..5 {
            y = apply_random_edit(&y, &alphabet);
        }

        let table = compute_table::<u16>(x, &y, Penalties::default());
        let (aligned_x, aligned_y) = trace_back_recursive(&table, [x, &y]);

        let actual_edits = unaligned_x_to_y(&aligned_x, &aligned_y);

        assert!(actual_edits.len() <= 5);
    }
}
