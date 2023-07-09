use distances::Number;

use super::Dataset;

// TODO: Remove pub from fields
pub struct VecVec<T: Send + Sync + Copy, U: Number> {
    pub name: String,
    pub data: Vec<T>,
    pub metric: fn(T, T) -> U,
    pub is_expensive: bool,
    pub indices: Vec<usize>,
    pub reordering: Option<Vec<usize>>,
}

impl<T: Send + Sync + Copy, U: Number> VecVec<T, U> {
    pub fn new(data: Vec<T>, metric: fn(T, T) -> U, name: String, is_expensive: bool) -> Self {
        assert_ne!(data.len(), 0, "Must have some instances in the data.");
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

impl<T: Send + Sync + Copy, U: Number> std::fmt::Debug for VecVec<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Space").field("name", &self.name).finish()
    }
}

impl<T: Send + Sync + Copy, U: Number> Dataset<T, U> for VecVec<T, U> {
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

    fn get(&self, index: usize) -> T {
        self.data[index]
    }

    fn metric(&self) -> fn(T, T) -> U {
        self.metric
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
    use rand::prelude::*;
    use symagen::random_data;

    use distances::vectors::euclidean_sq;

    use super::*;

    #[test]
    fn test_reordering_u32() {
        // 10 random 10 dimensional datasets reordered 10 times in 10 random ways
        let mut rng = rand::thread_rng();
        let name = "test".to_string();
        let cardinality = 10_000;

        for i in 0..10 {
            let dimensionality = 10;
            let reference_data = random_data::random_u32(cardinality, dimensionality, 0, 100_000, i);
            let reference_data = reference_data.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
            for _ in 0..10 {
                let mut dataset = VecVec::new(reference_data.clone(), euclidean_sq::<u32, u32>, name.clone(), false);
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
        let data: Vec<Vec<u32>> = (1..7).map(|x| vec![(x * 2) as u32]).collect();
        let data: Vec<&[u32]> = data.iter().map(|v| v.as_slice()).collect();
        let permutation = vec![1, 3, 4, 0, 5, 2];

        let mut dataset = VecVec::new(data, euclidean_sq::<u32, u32>, "test".to_string(), false);

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
