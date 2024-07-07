//! Augment an existing dataset by adding random points that lie near its manifold.

use distances::Number;
use rayon::prelude::*;

use crate::random_data;

/// Generate an augmented dataset from an existing dataset by adding random points
/// that are within a certain error distance of the original points.
///
/// # Arguments:
///
/// * `data`: the existing dataset
/// * `multiplier`: the number of new points to make per existing point
/// * `error`: the maximum euclidean distance from the original point that the
///            new points can be
#[must_use]
pub fn augment_data(data: &[Vec<f32>], multiplier: usize, error: f32) -> Vec<Vec<f32>> {
    let dimensionality = data[0].len();
    let dimensional_error = error / dimensionality.as_f32().sqrt();

    data.par_iter()
        .flat_map(|point| {
            let perturbations = random_data::random_tabular(
                multiplier,
                dimensionality,
                1.0 - dimensional_error,
                1.0 + dimensional_error,
                &mut rand::thread_rng(),
            );
            perturbations
                .into_iter()
                .map(|perturbation| {
                    point
                        .iter()
                        .zip(perturbation.iter())
                        .map(|(&x, &y)| x * y)
                        .collect::<Vec<_>>()
                })
                .chain(std::iter::once(point.clone()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use crate::augmentation::augment_data;
    use crate::random_data::random_tabular_seedable;

    #[test]
    fn tiny() {
        let data = vec![vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]];

        let augmented_data = augment_data(&data, 3, 0.5);
        assert_eq!(12, augmented_data.len());
    }

    #[test]
    fn big() {
        let data = random_tabular_seedable(10000, 20, 0.1, 100.1, 42);
        let augmented_data = augment_data(&data, 10, 0.2);
        assert_eq!(110_000, augmented_data.len());

        let augmented_data = augment_data(&data, 100, 0.2);
        assert_eq!(1_010_000, augmented_data.len());
    }
}
