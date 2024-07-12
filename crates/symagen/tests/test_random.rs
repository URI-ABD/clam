use std::collections::HashMap;

use distances::strings::{
    levenshtein_custom,
    needleman_wunsch::{compute_table, trace_back_recursive},
    unaligned_x_to_y, Penalties,
};
use symagen::random_edits::{apply_random_edit, are_we_there_yet, create_batch};

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

#[test]
fn random_edits() {
    let alphabet = vec!['N', 'A', 'J', 'I', 'B', 'P', 'E', 'R', 'S', 'T'];
    let x = "NAJIBEATSPEPPERS";

    let penalties = Penalties::new(0, 1, 1);

    let new_string = are_we_there_yet::<u16>(x, penalties, 10, &alphabet);
    let lev = levenshtein_custom(penalties);

    // Should fail, for sanity:
    assert_eq!(
        lev(x, &new_string),
        10,
        "Distance between {x} and {new_string} is not 10"
    );
}

#[test]
fn random_batch() {
    let seed_string = "ACGGTTTGCGTAACGGTTTGCGTAACGGTTTGCGTA";
    let alphabet = vec!['A', 'C', 'G', 'T'];

    let penalties = Penalties::new(0, 1, 1);

    let batch_size = 100;
    let batch = create_batch::<u16>(seed_string, penalties, 10, 15, &alphabet, batch_size);
    let mut strings: HashMap<String, usize> = HashMap::new();
    for n in batch.iter() {
        strings
            .entry(n.to_string())
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    assert!(
        strings.len() >= 98,
        "Batch is not diverse enough. Only {} unique strings",
        strings.len()
    );
}
