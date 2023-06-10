use crate::number::Number;

use super::Dataset;

pub struct VecVec<T: Number, U: Number> {
    name: String,
    data: Vec<Vec<T>>,
    metric: fn(&[T], &[T]) -> U,
    is_expensive: bool,
    indices: Vec<usize>,
    reordering: Option<Vec<usize>>,
}

impl<T: Number, U: Number> VecVec<T, U> {
    pub fn new(data: Vec<Vec<T>>, metric: fn(&[T], &[T]) -> U, name: String, is_expensive: bool) -> Self {
        assert_ne!(data.len(), 0, "Must have some instances in the data.");
        assert_ne!(data[0].len(), 0, "Must have some numbers in the instances.");
        let indices = (0..data.len()).collect();
        Self {
            name,
            data,
            metric,
            is_expensive,
            indices,
            reordering: None,
        }
    }
}

impl<T: Number, U: Number> std::fmt::Debug for VecVec<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Space").field("name", &self.name).finish()
    }
}

impl<T: Number, U: Number> Dataset<T, U> for VecVec<T, U> {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn cardinality(&self) -> usize {
        self.data.len()
    }

    fn dimensionality(&self) -> usize {
        self.data[0].len()
    }

    fn is_metric_expensive(&self) -> bool {
        self.is_expensive
    }

    fn indices(&self) -> &[usize] {
        &self.indices
    }

    fn one_to_one(&self, left: usize, right: usize) -> U {
        (self.metric)(&self.data[left], &self.data[right])
    }

    fn query_to_one(&self, query: &[T], index: usize) -> U {
        (self.metric)(query, &self.data[index])
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.data.swap(i, j);
    }

    fn set_reordered_indices(&mut self, indices: &[usize]) {
        self.reordering = Some(indices.iter().map(|&i| indices[i]).collect());
    }

    fn get_reordered_index(&self, i: usize) -> usize {
        self.reordering.as_ref().map(|indices| indices[i]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;
    use rand::prelude::*;

    use crate::distances;
    use crate::utils::helpers::{gen_data_f32, gen_data_u32};

    use super::*;

    #[test]
    fn test_space() {
        let data = vec![vec![1_f32, 2., 3.], vec![3., 3., 1.]];
        let metric = distances::f32::euclidean;
        let name = "test".to_string();
        let dataset = VecVec::new(data, metric, name, false);

        approx_eq!(f32, dataset.one_to_one(0, 0), 0.);
        approx_eq!(f32, dataset.one_to_one(0, 1), 3.);
        approx_eq!(f32, dataset.one_to_one(1, 0), 3.);
        approx_eq!(f32, dataset.one_to_one(1, 1), 0.);
    }

    #[test]
    fn test_reordering_u32() {
        // 10 random 10 dimensional datasets reordered 10 times in 10 random ways
        let mut rng = rand::thread_rng();
        let name = "test".to_string();
        let cardinality = 10_000;

        for i in 0..10 {
            let reference_data: Vec<Vec<u32>> = gen_data_u32(cardinality, 10, 0, 100_000, i);
            for _ in 0..10 {
                let mut dataset = VecVec::new(reference_data.clone(), distances::u32::euclidean, name.clone(), false);
                let mut new_indices = dataset.indices().to_vec();
                new_indices.shuffle(&mut rng);

                dataset.reorder(&new_indices);
                for i in 0..cardinality {
                    assert_eq!(dataset.data[i], reference_data[new_indices[i]]);
                }
            }
        }
    }

    #[test]
    fn test_reordering_f32() {
        // 10 random 10 dimensional datasets reordered 10 times in 10 random ways
        let mut rng = rand::thread_rng();
        let name = "test".to_string();
        let cardinality = 10_000;

        for i in 0..10 {
            let reference_data: Vec<Vec<f32>> = gen_data_f32(cardinality, 10, 0., 100_000., i);
            for _ in 0..10 {
                let mut dataset = VecVec::new(reference_data.clone(), distances::f32::euclidean, name.clone(), false);
                let mut new_indices = dataset.indices().to_vec();
                new_indices.shuffle(&mut rng);

                dataset.reorder(&new_indices);

                // Assert each element in each row vector have been reordered correctly
                for i in 0..cardinality {
                    for index in 0..dataset.data[i].len() {
                        let delta = dataset.data[i][index] - reference_data[new_indices[i]][index];
                        assert!(delta.abs() < std::f32::EPSILON);
                    }
                }
            }
        }
    }

    #[test]
    fn test_inverse_map() {
        let data: Vec<Vec<u32>> = (1..7).map(|x| vec![(x * 2) as u32]).collect();
        let permutation = vec![1, 3, 4, 0, 5, 2];

        let mut dataset = VecVec::new(data, distances::u32::euclidean, "test".to_string(), false);

        dataset.reorder(&permutation);

        assert_eq!(
            dataset.data,
            vec![vec![4], vec![8], vec![10], vec![2], vec![12], vec![6],]
        );

        assert_eq!(dataset.get_reordered_index(0), 3);
        assert_eq!(dataset.data[dataset.get_reordered_index(0)], vec![2]);

        assert_eq!(dataset.get_reordered_index(1), 0);
        assert_eq!(dataset.data[dataset.get_reordered_index(1)], vec![4]);

        assert_eq!(dataset.get_reordered_index(2), 5);
        assert_eq!(dataset.data[dataset.get_reordered_index(2)], vec![6]);

        assert_eq!(dataset.get_reordered_index(3), 1);
        assert_eq!(dataset.data[dataset.get_reordered_index(3)], vec![8]);

        assert_eq!(dataset.get_reordered_index(4), 2);
        assert_eq!(dataset.data[dataset.get_reordered_index(4)], vec![10]);

        assert_eq!(dataset.get_reordered_index(5), 4);
        assert_eq!(dataset.data[dataset.get_reordered_index(5)], vec![12]);
    }
}
