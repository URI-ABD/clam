//! Generating random data for experimentation.

use rand::prelude::*;

/// Generate random data for experimentation.
///
/// The data is generated under a normal distribution with the given mean and
/// standard deviation.
///
/// # Arguments
///
/// - `mean`: The mean of the normal distribution.
/// - `std`: The standard deviation of the normal distribution.
/// - `car`: The number of vectors to generate.
/// - `dim`: The dimensionality of the vectors.
/// - `seed`: The seed for the random number generator.
pub fn gen_random(mean: f32, std: f32, car: usize, dim: usize, seed: Option<u64>) -> Vec<Vec<f32>> {
    let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);
    (0..car).map(|_| gen_vec(mean, std, dim, &mut rng)).collect()
}

/// Generate a random vector with the given mean and standard deviation under a
/// normal distribution.
fn gen_vec(mean: f32, std: f32, dim: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>().mul_add(std, mean)).collect()
}
