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

/// Generate a single point (in Cartesian coordinates) from a uniform distribution
/// inside an n-dimensional ball of given radius.
///
/// This function produces points in a uniform distribution inside the n-ball, and does
/// so in linear time in the dimensionality of the ball. The algorithm is based on the
/// method described in [this wikipedia article](https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates).
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
pub fn n_ball<R: Rng>(dim: usize, radius: f64, rng: &mut R) -> Vec<f64> {
    // sample random angles from 0 to 2π for the last angle and from 0 to π for the other angles.
    let angles = {
        let mut angles = (0..dim).map(|_| f64::next_random(rng) * PI).collect::<Vec<_>>();
        angles[dim - 1] *= 2.;
        angles
    };

    // The `scan` method is used to accumulate the product of the sine of the angles.
    let sine_products = angles.iter().scan(1., |sine_product, &x| {
        // each iteration, we'll multiply the state by the element ...
        *sine_product *= f64::sin(x);
        Some(*sine_product)
    });

    let cosines = angles.iter().map(|&x| f64::cos(x));

    // sample a random radius value from 0 to the given radius
    let r = radius * f64::next_random(rng);

    sine_products
        .zip(cosines)
        .map(|(sine_product, cosine)| r * sine_product * cosine)
        .collect()
}
