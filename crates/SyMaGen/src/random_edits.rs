//! Generate random strings for use in benchmarks, tests, and compression experiments.

use distances::number::UInt;
use distances::strings::levenshtein_custom;
use distances::strings::needleman_wunsch::apply_edits;
use distances::strings::Edit;
use distances::strings::Penalties;

/// Generates (but does not apply) a random edit to a given string.
/// The character (if applicable) for the edit is a random character from the given alphabet.
///
/// # Arguments
///
/// * `string`: The string to apply the edit to.
/// * `alphabet`: The alphabet to choose the character from.
///
/// # Returns
///
/// A random edit (Deletion, Insertion, or Substitution) based on the given string and alphabet.
#[must_use]
pub fn generate_random_edit(string: &str, alphabet: &[char]) -> Edit {
    let edit_type = rand::random::<u8>() % 3;
    let length = string.len();
    let char = alphabet[rand::random::<usize>() % alphabet.len()];

    match edit_type {
        0 => {
            let index = rand::random::<usize>() % (length + 1);
            Edit::Ins(index, char)
        }
        1 => Edit::Del(rand::random::<usize>() % length),
        2 => Edit::Sub(rand::random::<usize>() % length, char),
        _ => unreachable!(),
    }
}

/// Applies a random edit to a given string.
///
///
/// # Arguments
///
/// * `string`: The string to apply the edit to.
/// * `alphabet`: The alphabet to choose the character from.
///
/// # Returns
///
/// A string with a random edit applied.
#[must_use]
pub fn apply_random_edit(string: &str, alphabet: &[char]) -> String {
    let random_edit = generate_random_edit(string, alphabet);

    apply_edits(string, &[random_edit])
}

/// Randomly applies edits to a string until the distance between the new string and the seed string is greater than or equal to the target distance.
///
/// # Arguments
///
/// * `seed_string`: The string to start with.
/// * `penalties`: The penalties for the edit operations.
/// * `target_distance`: The desired distance between the seed and the new string.
/// * `alphabet`: The alphabet to choose characters from.
///
/// # Returns
///
/// A randomly generated string with a distance greater than or equal to the target distance from the seed string.
pub fn are_we_there_yet<U: UInt>(
    seed_string: &str,
    penalties: Penalties<U>,
    target_distance: U,
    alphabet: &[char],
) -> String {
    let mut new_string = seed_string.to_string();
    let mut distance = U::zero();
    let lev = levenshtein_custom(penalties);

    while distance < target_distance {
        let edit = generate_random_edit(&new_string, alphabet);
        new_string = apply_edits(&new_string, &[edit]);
        distance = lev(seed_string, &new_string);
    }

    new_string
}

/// Generates a batch of strings with random edits applied to a seed string.
///
/// # Arguments
///
/// * `seed_string`: The string to start with.
/// * `penalties`: The penalties for the edit operations.
/// * `target_distance`: The desired distance between the seed and the new string.
/// * `alphabet`: The alphabet to choose characters from.
/// * `batch_size`: The number of strings to generate.
///
/// # Returns
///
/// A vector of randomly generated strings with a distance slightly greater than or equal to the target distance from the seed string.
pub fn create_batch<U: UInt>(
    seed_string: &str,
    penalties: Penalties<U>,
    target_distance: U,
    alphabet: &[char],
    batch_size: usize,
) -> Vec<String> {
    (0..batch_size)
        .map(|_| are_we_there_yet(seed_string, penalties, target_distance, alphabet))
        .collect()
}
