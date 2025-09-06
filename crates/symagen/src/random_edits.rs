//! Generate random strings for use in benchmarks, tests, and compression experiments.

use distances::{
    Number,
    number::UInt,
    strings::{Edit, Penalties, levenshtein_custom, needleman_wunsch::apply_edits},
};
use rayon::prelude::*;

/// Generates a random string of a given length from a given alphabet.
///
/// # Arguments
///
/// * `length`: The length of the string to generate.
/// * `alphabet`: The alphabet to choose characters from.
///
/// # Returns
///
/// A random string of the given length from the given alphabet.
#[must_use]
pub fn generate_random_string(length: usize, alphabet: &[char]) -> String {
    (0..length).map(|_| alphabet[rand::random::<u64>().as_usize() % alphabet.len()]).collect()
}

/// Generates (but does not apply) a random edit to a given string.
/// The character (if applicable) for the edit is a random character from the given alphabet.
///
/// # Arguments
///
/// * `string`: The string to apply the edit to.
/// * `alphabet`: The alphabet to choose the character from.
/// * `min_len`: The minimum length of the output string.
/// * `max_len`: The maximum length of the output string.
///
/// # Returns
///
/// A random edit (Deletion, Insertion, or Substitution) based on the given string and alphabet.
#[must_use]
pub fn generate_random_edit(string: &str, alphabet: &[char], min_len: usize, max_len: usize) -> Edit {
    let length = string.len();

    let edit_type = if length == min_len {
        // If the string is at the minimum length, we can only insert or substitute
        rand::random::<u8>() % 2
    } else if length == max_len {
        // If the string is at the maximum length, we can only delete or substitute
        rand::random::<u8>() % 2 + 1
    } else {
        // Otherwise, we can do any edit
        rand::random::<u8>() % 3
    };

    let char = alphabet[rand::random::<u64>().as_usize() % alphabet.len()];
    match edit_type {
        0 => {
            let index = rand::random::<u64>().as_usize() % (length + 1);
            Edit::Ins(index, char)
        }
        1 => Edit::Sub(rand::random::<u64>().as_usize() % length, char),
        2 => Edit::Del(rand::random::<u64>().as_usize() % length),
        _ => unreachable!(),
    }
}

/// Randomly applies edits to a string until the distance between the new string and the seed string is greater than or equal to the target distance.
///
/// # Arguments
///
/// * `seed_string`: The string to start with.
/// * `penalties`: The penalties for the edit operations.
/// * `alphabet`: The alphabet to choose characters from.
/// * `target_distance`: The desired distance between the seed and the new string.
/// * `min_len`: The minimum length of the output string.
/// * `max_len`: The maximum length of the output string.
///
/// # Returns
///
/// A randomly generated string with a distance greater than or equal to the target distance from the seed string.
pub fn are_we_there_yet<U: UInt>(seed_string: &str, penalties: Penalties<U>, alphabet: &[char], target_distance: U, len_delta: usize) -> String {
    let mut new_string = seed_string.to_string();
    let mut distance = U::ZERO;
    let lev = levenshtein_custom(penalties);
    let (min_len, max_len) = (seed_string.len() - len_delta, seed_string.len() + len_delta);

    while distance < target_distance {
        let edit = generate_random_edit(&new_string, alphabet, min_len, max_len);
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
    min_distance: U,
    max_distance: U,
    alphabet: &[char],
    batch_size: usize,
    len_delta: usize,
) -> Vec<String> {
    let mut others = (1..batch_size)
        .into_par_iter()
        .map(|_| {
            // Randomly sample a distance between 1 and `target_distance`, inclusive
            let delta = max_distance - min_distance + U::ONE;
            let d = min_distance.as_u64() + rand::random::<u64>() % delta.as_u64();
            let target_distance = U::from(d);
            are_we_there_yet(seed_string, penalties, alphabet, target_distance, len_delta)
        })
        .collect::<Vec<_>>();
    others.push(seed_string.to_string());
    others
}

#[must_use]
/// Generates a random batch of strings in distinct clumps.
///
/// # Arguments
///
/// * `seed_string`: A baseline string to serve as the center of one clump.
/// * `penalties`: The penalties for the edit operations.
/// * `alphabet`: The alphabet to choose characters from.
/// * `num_clumps`: The number of clumps to generate.
/// * `clump_size`: The number of strings in each clump.
/// * `clump_radius`: The target distance from the seed string for each clump.
/// * `inter_clump_distance_range`: The range of distances between the centers of each clump.
/// * `len_delta`: The maximum difference in length between the seed string and the generated strings.
///
/// # Returns
///
/// - The seed string for each clump.
/// - The strings in each clump.
#[allow(clippy::too_many_arguments)]
pub fn generate_clumped_data<U: UInt>(
    seed_string: &str,
    penalties: Penalties<U>,
    alphabet: &[char],
    num_clumps: usize,
    clump_size: usize,
    clump_radius: U,
    inter_clump_distance_range: (U, U),
    len_delta: usize,
) -> Vec<(String, String)> {
    // TODO(Morgan): add min length and max length for strings as inputs here

    // Vector of seed strings for each clump (can think of as the ``center''s of each clump)

    // TODO(Morgan): change 10 to input parameter inter-clump distance
    let (min_distance, max_distance) = inter_clump_distance_range;
    let clump_seeds = create_batch(seed_string, penalties, min_distance, max_distance, alphabet, num_clumps, len_delta);

    // Generate the clumps
    clump_seeds
        .iter()
        .enumerate()
        .flat_map(|(i, seed)| {
            create_batch(seed, penalties, U::ONE, clump_radius, alphabet, clump_size, len_delta)
                .into_iter()
                .enumerate()
                .map(move |(j, string)| (format!("{i}x{j}"), string))
        })
        .collect()
}
