//! A dataset of a Vec of instances.

use distances::Number;

use crate::Dataset;

/// A `Dataset` of a `Vec` of instances.
///
/// This may be used for any data that can fit in memory. It is not recommended for large datasets.
///
/// # Type Parameters
///
/// - `T`: The type of the instances in the `Dataset`.
/// - `U`: The type of the distance values between instances.
pub struct VecDataset<T: Send + Sync, U: Number> {
    /// The name of the dataset.
    pub(crate) name: String,
    /// The data of the dataset.
    pub(crate) data: Vec<T>,
    /// The metric of the dataset.
    pub(crate) metric: fn(&T, &T) -> U,
    /// Whether the metric is expensive to compute.
    pub(crate) is_expensive: bool,
    /// The indices of the dataset.
    pub(crate) indices: Vec<usize>,
    /// The reordering of the dataset after building the tree.
    pub(crate) reordering: Option<Vec<usize>>,
}

impl<T: Send + Sync, U: Number> VecDataset<T, U> {
    /// Creates a new dataset.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the dataset.
    /// * `data`: The vector of instances.
    /// * `metric`: The metric for computing distances between instances.
    /// * `is_expensive`: Whether the metric is expensive to compute.
    pub fn new(name: String, data: Vec<T>, metric: fn(&T, &T) -> U, is_expensive: bool) -> Self {
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

impl<T: Send + Sync, U: Number> std::fmt::Debug for VecDataset<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Space").field("name", &self.name).finish()
    }
}

impl<T: Send + Sync + Clone, U: Number> Dataset<T, U> for VecDataset<T, U> {
    fn name(&self) -> &str {
        &self.name
    }

    fn cardinality(&self) -> usize {
        self.data.len()
    }

    fn is_metric_expensive(&self) -> bool {
        self.is_expensive
    }

    fn indices(&self) -> &[usize] {
        &self.indices
    }

    fn get(&self, index: usize) -> Box<T> {
        Box::new(self.data[index].clone())
    }

    // fn metric(&self) -> fn(&T, &T) -> U{
    //     self.metric
    // }

    fn swap(&mut self, i: usize, j: usize) {
        self.data.swap(i, j);
    }

    fn set_reordered_indices(&mut self, indices: &[usize]) {
        self.reordering = Some(indices.iter().map(|&i| indices[i]).collect());
    }

    fn get_reordered_index(&self, i: usize) -> Option<usize> {
        self.reordering.as_ref().map(|indices| indices[i])
    }

    fn query_to_one(&self, query: &T, index: usize) -> U {
        (self.metric)(query, &self.data[index])
    }

    fn one_to_one(&self, left: usize, right: usize) -> U {
        (self.metric)(&self.data[left], &self.data[right])
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use symagen::random_data;

    use super::*;

    fn metric(a: &Vec<u32>, b: &Vec<u32>) -> u32 {
        distances::vectors::euclidean_sq(a, b)
    }

    #[test]
    fn test_reordering_u32() {
        // 10 random 10 dimensional datasets reordered 10 times in 10 random ways
        let mut rng = rand::thread_rng();
        let name = "test".to_string();
        let cardinality = 10_000;

        for i in 0..10 {
            let dimensionality = 10;
            let reference_data = random_data::random_u32(cardinality, dimensionality, 0, 100_000, i);
            for _ in 0..10 {
                let mut dataset = VecDataset::new(name.clone(), reference_data.clone(), metric, false);
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
    fn test_inverse_map() {
        let data: Vec<Vec<u32>> = (1_u32..7).map(|x| vec![x * 2]).collect();
        let permutation = vec![1, 3, 4, 0, 5, 2];

        let mut dataset = VecDataset::new("test".to_string(), data, metric, false);

        dataset.reorder(&permutation);

        assert_eq!(
            dataset.data,
            vec![vec![4], vec![8], vec![10], vec![2], vec![12], vec![6],]
        );

        assert_eq!(dataset.get_reordered_index(0), Some(3));
        assert_eq!(
            dataset.get_reordered_index(0).map(|i| dataset.data[i].clone()),
            Some(vec![2])
        );

        assert_eq!(dataset.get_reordered_index(1), Some(0));
        assert_eq!(
            dataset.get_reordered_index(1).map(|i| dataset.data[i].clone()),
            Some(vec![4])
        );

        assert_eq!(dataset.get_reordered_index(2), Some(5));
        assert_eq!(
            dataset.get_reordered_index(2).map(|i| dataset.data[i].clone()),
            Some(vec![6])
        );

        assert_eq!(dataset.get_reordered_index(3), Some(1));
        assert_eq!(
            dataset.get_reordered_index(3).map(|i| dataset.data[i].clone()),
            Some(vec![8])
        );

        assert_eq!(dataset.get_reordered_index(4), Some(2));
        assert_eq!(
            dataset.get_reordered_index(4).map(|i| dataset.data[i].clone()),
            Some(vec![10])
        );

        assert_eq!(dataset.get_reordered_index(5), Some(4));
        assert_eq!(
            dataset.get_reordered_index(5).map(|i| dataset.data[i].clone()),
            Some(vec![12])
        );
    }
}
