//! Generate random data for use in benchmarks and tests.

use rand::prelude::*;

/// Generate a randomized tabular dataset for use in benchmarks and tests.
///
/// # Arguments:
///
/// * `cardinality`: number of points to generate.
/// * `dimensionality`: dimensionality of points to generate.
/// * `min_val`: of each axis in the hypercube
/// * `max_val`: of each axis in the hypercube
/// * `seed`: for the random number generator
#[must_use]
pub fn random_u8(cardinality: usize, dimensionality: usize, min_val: u8, max_val: u8, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..cardinality)
        .map(|_| (0..dimensionality).map(|_| rng.gen_range(min_val..=max_val)).collect())
        .collect()
}

/// Generate a randomized tabular dataset for use in benchmarks and tests.
///
/// # Arguments:
///
/// * `cardinality`: number of points to generate.
/// * `dimensionality`: dimensionality of points to generate.
/// * `min_val`: of each axis in the hypercube
/// * `max_val`: of each axis in the hypercube
/// * `seed`: for the random number generator
#[must_use]
pub fn random_f32(cardinality: usize, dimensionality: usize, min_val: f32, max_val: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..cardinality)
        .map(|_| (0..dimensionality).map(|_| rng.gen_range(min_val..=max_val)).collect())
        .collect()
}

/// Generate a randomized tabular dataset for use in benchmarks and tests.
///
/// # Arguments:
///
/// * `cardinality`: number of points to generate.
/// * `dimensionality`: dimensionality of points to generate.
/// * `min_val`: of each axis in the hypercube
/// * `max_val`: of each axis in the hypercube
/// * `seed`: for the random number generator
#[must_use]
pub fn random_f64(cardinality: usize, dimensionality: usize, min_val: f64, max_val: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..cardinality)
        .map(|_| (0..dimensionality).map(|_| rng.gen_range(min_val..=max_val)).collect())
        .collect()
}

/// Generate a randomized tabular dataset for use in benchmarks and tests.
///
/// # Arguments:
///
/// * `cardinality`: number of points to generate.
/// * `dimensionality`: dimensionality of points to generate.
/// * `min_val`: of each axis in the hypercube
/// * `max_val`: of each axis in the hypercube
/// * `seed`: for the random number generator
#[must_use]
pub fn random_u32(cardinality: usize, dimensionality: usize, min_val: u32, max_val: u32, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..cardinality)
        .map(|_| (0..dimensionality).map(|_| rng.gen_range(min_val..=max_val)).collect())
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
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..cardinality)
        .map(|_| {
            let len = rng.gen_range(min_len..=max_len);
            (0..len)
                .map(|_| alphabet[rng.gen_range(0..alphabet.len())])
                .collect::<String>()
        })
        .collect()
}
