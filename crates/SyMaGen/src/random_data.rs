//! Generate random data for use in benchmarks and tests.

use distances::number::Number;
use rand::prelude::*;

/// The mathematical constant π.
pub const PI: f64 = std::f64::consts::PI;

/// Generate a randomized tabular dataset for use in benchmarks and tests with a
/// given seed.
///
/// This uses the `rand` crate's `StdRng` as the random number generator.
///
/// # Arguments:
///
/// * `cardinality`: number of points to generate.
/// * `dimensionality`: dimensionality of points to generate.
/// * `min_val`: of each axis in the hypercube.
/// * `max_val`: of each axis in the hypercube.
/// * `seed`: for the random number generator.
#[must_use]
pub fn random_tabular_seedable<T: Number>(
    cardinality: usize,
    dimensionality: usize,
    min_val: T,
    max_val: T,
    seed: u64,
) -> Vec<Vec<T>> {
    random_tabular(
        cardinality,
        dimensionality,
        min_val,
        max_val,
        &mut rand::rngs::StdRng::seed_from_u64(seed),
    )
}

/// Generate a randomized tabular dataset for use in benchmarks and tests.
///
/// # Arguments:
///
/// * `cardinality`: number of points to generate.
/// * `dimensionality`: dimensionality of points to generate.
/// * `min_val`: of each axis in the hypercube.
/// * `max_val`: of each axis in the hypercube.
/// * `rng`: random number generator.
#[must_use]
pub fn random_tabular<T: Number, R: Rng>(
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

/// Generate a single point (in Cartesian coordinates) from a uniform distribution over an n-dimensional ball of given radius.
///
/// # Arguments:
///
/// * `dim`: dimensionality of the point to generate
/// * `radius`: radius of the ball
/// * `rng`: random number generator
///
/// # Returns:
///
/// * `Vec<T>`: the generated point
pub fn n_ball<R: Rng>(_dim: usize, radius: f64, _rng: &mut R) -> Vec<f64> {
    // let _angles: Vec<f64> = (0..dim).map(|_| f64::next_random(rng) % f64::from(PI)).collect();
    // let angles = random_tabular(1,dim, 0, 2.*PI,  &mut 42);
    let angles = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let sine_products = angles.iter().scan(1., |sine_product, &x| {
        // each iteration, we'll multiply the state by the element ...
        *sine_product *= f64::sin(x);
        Some(*sine_product)
    });

    let cosines = angles.iter().map(|&x| f64::cos(x)).collect::<Vec<_>>();

    let cartesian_points = sine_products
        .zip(cosines.iter())
        .map(|(sine_product, &cosine)| sine_product * cosine * radius)
        .collect::<Vec<_>>();

    cartesian_points
}
