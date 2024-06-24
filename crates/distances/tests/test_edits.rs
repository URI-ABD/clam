use distances::strings::{Edit, _x_to_y, apply_edits, unaligned_x_to_y};

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
fn test_apply_edits() {
    let x = "ACBAAB";
    let edits: Vec<Edit> = vec![
        Edit::Ins(1, 'A'),
        Edit::Sub(3, 'A'),
        Edit::Del(4),
        Edit::Sub(5, 'C'),
    ];

    let actual = apply_edits(x, &edits);

    let expected = "AACAAC";

    assert_eq!(actual, expected);
}
