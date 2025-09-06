#![allow(missing_docs)]

use std::collections::HashMap;

use distances::strings::{Penalties, levenshtein_custom};
use symagen::random_edits::{are_we_there_yet, create_batch};

#[test]
fn random_edits() {
    let alphabet = vec!['N', 'A', 'J', 'I', 'B', 'P', 'E', 'R', 'S', 'T'];
    let seed_string = "NAJIBEATSPEPPERS";

    let penalties = Penalties::new(0, 1, 1);
    let target_distance = 10;
    let len_delta = 3;

    let new_string = are_we_there_yet::<u16>(seed_string, penalties, &alphabet, target_distance, len_delta);
    let lev = levenshtein_custom(penalties);

    // Should fail, for sanity:
    assert_eq!(lev(seed_string, &new_string), 10, "Distance between {seed_string} and {new_string} is not 10");
}

#[test]
fn random_batch() {
    let seed_string = "ACGGTTTGCGTAACGGTTTGCGTAACGGTTTGCGTA";
    let alphabet = vec!['A', 'C', 'G', 'T'];

    let penalties = Penalties::new(0, 1, 1);
    let len_delta = 5;

    let batch_size = 100;
    let batch = create_batch::<u16>(seed_string, penalties, 10, 15, &alphabet, batch_size, len_delta);
    let mut strings: HashMap<String, usize> = HashMap::new();
    for n in batch.iter() {
        strings.entry(n.to_string()).and_modify(|count| *count += 1).or_insert(1);
    }

    assert!(strings.len() >= 98, "Batch is not diverse enough. Only {} unique strings", strings.len());
}
