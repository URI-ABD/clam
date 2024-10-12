#[test]
fn test_levenshtein() {
    for d in 2..=4 {
        let len = 10_usize.pow(d);
        let vecs = symagen::random_data::random_string(2, len, len, "ATCGN", 42);
        let (x, y) = (&vecs[0], &vecs[1]);

        let dist = distances::strings::levenshtein::<usize>(x, y);
        let szla = stringzilla::sz::edit_distance(x, y);

        assert_eq!(dist, szla);
    }
}
