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
pub fn random_u32(cardinality: usize, dimensionality: usize, min_val: u32, max_val: u32, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..cardinality)
        .map(|_| (0..dimensionality).map(|_| rng.gen_range(min_val..=max_val)).collect())
        .collect()
}
