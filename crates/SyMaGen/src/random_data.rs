//! Generate random data for use in benchmarks and tests.

use distances::number::{Float, Int};
use rand::prelude::*;

/// Generate a randomized tabular dataset of integers for use in benchmarks and tests.
///
/// # Arguments:
///
/// * `cardinality`: number of points to generate.
/// * `dimensionality`: dimensionality of points to generate.
/// * `min_val`: of each axis in the hypercube.
/// * `max_val`: of each axis in the hypercube.
/// * `rng`: random number generator.
#[must_use]
pub fn random_tabular_integers<T: Int, R: Rng>(
    cardinality: usize,
    dimensionality: usize,
    min_val: T,
    max_val: T,
    rng: &mut R,
) -> Vec<Vec<T>> {
    let diff = max_val - min_val;
    (0..cardinality)
        .map(|_| {
            (0..dimensionality)
                .map(|_| min_val + T::next_random(rng) % diff)
                .collect()
        })
        .collect()
}

/// Generate a randomized tabular dataset of floats for use in benchmarks and tests.
///
/// # Arguments:
///
/// * `cardinality`: number of points to generate.
/// * `dimensionality`: dimensionality of points to generate.
/// * `min_val`: of each axis in the hypercube.
/// * `max_val`: of each axis in the hypercube.
/// * `rng`: random number generator.
#[must_use]
pub fn random_tabular_floats<T: Float, R: Rng>(
    cardinality: usize,
    dimensionality: usize,
    min_val: T,
    max_val: T,
    rng: &mut R,
) -> Vec<Vec<T>> {
    let diff = max_val - min_val;
    (0..cardinality)
        .map(|_| {
            (0..dimensionality)
                .map(|_| min_val + T::next_random(rng) * diff)
                .collect()
        })
        .collect()
}

/// Generate a randomized dataset of string sequences.
///
/// # Arguments:
///
/// * `cardinality`: number of strings to generate.
/// * `min_len`: minimum length of any string
/// * `max_len`: maximum length of any string
/// * `alphabet`: the alphabet from which to draw characters
/// * `seed`: for the random number generator
#[must_use]
pub fn random_string(cardinality: usize, min_len: usize, max_len: usize, alphabet: &str, seed: u64) -> Vec<String> {
    let alphabet = alphabet.chars().collect::<Vec<_>>();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..cardinality)
        .map(|_| {
            let len = rng.gen_range(min_len..=max_len);
            (0..len)
                .map(|_| alphabet[rng.gen_range(0..alphabet.len())])
                .collect::<String>()
        })
        .collect()
}
