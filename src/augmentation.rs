//! Augment an existing dataset by adding random points that lie near its manifold.

use distances::Number;
use rand::prelude::*;

#[must_use]
/// Generate an augmented dataset from an existing dataset by adding random points
/// that are within a certain error distance of the original points.
///
/// # Arguments:
///
/// * `data`: the existing dataset
/// * `multiplier`: the number of new points to make per existing point
/// * `error`: the maximum distance from the original point that the new points can be
pub fn augment_data(data: &[Vec<f32>], multiplier: usize, error: f32) -> Vec<Vec<f32>> {
    let adjusted_error = error / data[0].len().as_f32().sqrt();

    let augmented_dataset = data.iter().flat_map(|point| {
        let mut augmented_points = vec![point.clone()];
        for _i in 1..=multiplier {
            let mut augmented_point = point.clone();

            for entry in augmented_point.iter_mut().take(point.len()) {
                let random_error = rand::thread_rng().gen_range(-adjusted_error..adjusted_error);
                *entry += random_error;
            }
            augmented_points.push(augmented_point);
        }
        augmented_points
    });

    augmented_dataset.collect()
}

#[cfg(test)]
mod tests {
    use crate::augmentation::augment_data;
    use crate::random_data::random_f32;

    #[test]
    fn tiny() {
        let data = vec![vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]];

        let augmented_data = augment_data(&data, 3, 0.5);
        assert_eq!(12, augmented_data.len());
    }

    #[test]
    fn big() {
        let data = random_f32(10000, 20, 0.1, 100.1, 42);
        let augmented_data = augment_data(&data, 10, 0.2);
        assert_eq!(110000, augmented_data.len());

        let augmented_data = augment_data(&data, 100, 0.2);
        assert_eq!(1010000, augmented_data.len());
    }
}
