use distances::strings::Edit;
use distances::strings::_x_to_y;
use distances::strings::needleman_wunsch::apply_edits;
use distances::strings::unaligned_x_to_y;

#[allow(dead_code)]
/// Applies a random edit to a given string. The edit is either an insertion, deletion, or substitution.
/// If the edit is a deletion or substitution, the index can be any index in the string. If the
/// edit is an insertion, the index can be the index before any character in the string, an index
/// between characters in the string, or the index after the last character in the string.
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
fn apply_random_edit(string: &str, alphabet: Vec<char>) -> String {
    let edit_type = rand::random::<u8>() % 3;
    let length = string.len();
    let char = alphabet[rand::random::<usize>() % alphabet.len()];

    match edit_type {
        0 => {
            let index = rand::random::<usize>() % (length + 2);
            if index < length + 2 {
                apply_edits(string, &[Edit::Ins(index, char)])
            } else {
                char.to_string() + string
            }
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
