pub fn levenshtein(a: &str, b: &str) -> usize {
    let (len_a, len_b) = (a.chars().count(), b.chars().count());

    if len_a == 0 {
        // handle special case of 0 length
        len_b
    } else if len_b == 0 {
        // handle special case of 0 length
        len_a
    } else if len_a < len_b {
        // require len_a < len_b
        levenshtein(b, a)
    } else {
        let len_b = len_b + 1;

        // initialize DP table for string b
        let mut cur: Vec<usize> = (0..len_b).collect();

        // calculate edit distance
        for (i, ca) in a.chars().enumerate() {
            // get first column for this row
            let mut pre = cur[0];
            cur[0] = i + 1;
            for (j, cb) in b.chars().enumerate() {
                let tmp = cur[j + 1];
                cur[j + 1] = std::cmp::min(
                    // deletion
                    tmp + 1,
                    std::cmp::min(
                        // insertion
                        cur[j] + 1,
                        // match or substitution
                        pre + if ca == cb { 0 } else { 1 },
                    ),
                );
                pre = tmp;
            }
        }
        cur[len_b - 1]
    }
}
