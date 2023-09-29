//! A dataset of a Vec of instances.

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use distances::Number;
use rand::prelude::*;

use crate::Dataset;

/// A `Dataset` of a `Vec` of instances.
///
/// This may be used for any data that can fit in memory. It is not recommended for large datasets.
///
/// # Type Parameters
///
/// - `T`: The type of the instances in the `Dataset`.
/// - `U`: The type of the distance values between instances.
pub struct VecDataset<T: Send + Sync + Copy, U: Number> {
    /// The name of the dataset.
    pub(crate) name: String,
    /// The data of the dataset.
    pub(crate) data: Vec<T>,
    /// The metric of the dataset.
    pub(crate) metric: fn(T, T) -> U,
    /// Whether the metric is expensive to compute.
    pub(crate) is_expensive: bool,
    /// The indices of the dataset.
    pub(crate) indices: Vec<usize>,
    /// The reordering of the dataset after building the tree.
    pub(crate) reordering: Option<Vec<usize>>,
}

impl<T: Send + Sync + Copy, U: Number> VecDataset<T, U> {
    /// Creates a new dataset.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the dataset.
    /// * `data`: The vector of instances.
    /// * `metric`: The metric for computing distances between instances.
    /// * `is_expensive`: Whether the metric is expensive to compute.
    pub fn new(name: String, data: Vec<T>, metric: fn(T, T) -> U, is_expensive: bool) -> Self {
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

impl<'a, V: Number, U: Number> VecDataset<&'a [V], U> {
    /// Saves a numeric dataset to a specified path in binary format
    ///
    /// # Errors
    /// Errors on file write issues or if a reordered index does not exist
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut handle = File::create(path).map_err(|err| format!("{err}"))?;

        // Write header (Basic protection against reading bad data)
        handle.write_all(b"VEC2D").map_err(|err| format!("{err}"))?;

        // Write dataset name
        handle.write_all(self.name.as_bytes()).map_err(|err| format!("{err}"))?;

        // Write null terminator
        handle.write(&[0]).map_err(|err| format!("{err}"))?;

        // Write cardinality
        let cardinality = self.data.len();
        let cardinality_bytes = cardinality.to_le_bytes();
        handle.write_all(&cardinality_bytes).map_err(|err| format!("{err}"))?;

        // Write dimensionality
        let dimensionality = self.data[0].len();
        let dimensionality_bytes = dimensionality.to_le_bytes();
        handle
            .write_all(&dimensionality_bytes)
            .map_err(|err| format!("{err}"))?;

        // Write the type name for V
        handle
            .write_all(V::type_name().as_bytes())
            .map_err(|err| format!("{err}"))?;

        // Write null terminator
        handle.write(&[0]).map_err(|err| format!("{err}"))?;

        // Write individual vectors
        for row in &self.data {
            for column in *row {
                let column_bytes = column.to_le_bytes();
                handle.write_all(&column_bytes).map_err(|err| format!("{err}"))?;
            }
        }

        // If the reordering map exists, write it and preprend a 1 for the flag byte. Otherwise write
        // a 0 byte
        if let Some(map) = &self.reordering {
            // Write one byte corresponding to if the reordering map exists
            handle.write(&[1_u8]).map_err(|err| format!("{err}"))?;

            for idx in map {
                let fullwidth = *idx as u64;

                handle
                    .write_all(&fullwidth.to_le_bytes())
                    .map_err(|err| format!("{err}"))?;
            }
        } else {
            handle.write(&[0]).map_err(|err| format!("{err}"))?;
        }

        Ok(())
    }

    /// Reads a saved numeric dataset from a specified path. The buffer passed in is the actual place where data is stored
    /// because `Vec2d` only holds references
    ///
    /// # Errors
    /// Errors on read write issues
    pub fn load(
        path: &Path,
        metric: fn(&[V], &[V]) -> U,
        is_expensive: bool,
        buffer: &'a mut Vec<Vec<V>>,
    ) -> Result<Self, String> {
        let mut handle = File::open(path).map_err(|err| format!("{err}"))?;

        // Decode the header (Assert the header is equal to 'VEC2D')
        let mut header_buf = [0_u8; 5];
        handle.read_exact(&mut header_buf).map_err(|err| format!("{err}"))?;
        let found: String = header_buf.map(|c| c as char).iter().collect();
        // If the header didn't match, return an error
        if found != "VEC2D" {
            return Err("Invalid header".to_string());
        }

        // Get the dataset's name
        let mut name = String::new();
        let mut char_buf = [0];
        handle.read_exact(&mut char_buf).map_err(|err| format!("{err}"))?;

        while char_buf[0] != 0 {
            name.push(char_buf[0] as char);
            handle.read_exact(&mut char_buf).map_err(|err| format!("{err}"))?;
        }

        // Get the cardinality and dimensionality
        let mut cardinality_buf = [0_u8; 8];
        let mut dimensionality_buf = [0_u8; 8];

        // Read in the cardinality and dimensionality
        handle
            .read_exact(&mut cardinality_buf)
            .map_err(|err| format!("{err}"))?;
        handle
            .read_exact(&mut dimensionality_buf)
            .map_err(|err| format!("{err}"))?;

        // Convert them to usize
        let cardinality = usize::from_le_bytes(cardinality_buf);
        let dimensionality = usize::from_le_bytes(dimensionality_buf);

        // Now read the type name
        let mut type_name = String::new();
        let mut char_buf = [0];
        handle.read_exact(&mut char_buf).map_err(|err| format!("{err}"))?;

        while char_buf[0] != 0 {
            type_name.push(char_buf[0] as char);
            handle.read_exact(&mut char_buf).map_err(|err| format!("{err}"))?;
        }

        // If the given type does not match the read type, error out
        if type_name != V::type_name() {
            return Err(format!(
                "Invalid type. File has data of type {} but dataset was constructed with type {}",
                type_name,
                V::type_name()
            ));
        }

        // Get each data point
        let type_size = V::num_bytes();

        let mut value_buf = vec![0_u8; type_size];
        *buffer = Vec::with_capacity(cardinality);

        for _ in 0..cardinality {
            let mut datum: Vec<V> = Vec::with_capacity(dimensionality);
            for _ in 0..dimensionality {
                handle.read_exact(&mut value_buf).map_err(|err| format!("{err}"))?;
                let index = V::from_le_bytes(&value_buf);
                datum.push(index);
            }
            buffer.push(datum);
        }

        let data: Vec<&[V]> = buffer.iter().map(Vec::as_slice).collect();

        // Now, check if there's a reordering map
        let mut bool_buf = [0_u8];
        handle.read_exact(&mut bool_buf).map_err(|err| format!("{err}"))?;

        // If we have a reordering map, read it
        let reordering: Option<Vec<usize>> = if bool_buf[0] == 1 {
            // Cardinality of the reordering map is equal to the cardinality of the datset
            // so we read 8 bytes (one u64) *cardinality* times.
            let mut map: Vec<usize> = Vec::with_capacity(cardinality);

            let mut usize_buf = [0_u8; 8];
            for _ in 0..cardinality {
                handle.read_exact(&mut usize_buf).map_err(|err| format!("{err}"))?;

                let mapping = usize::from_le_bytes(usize_buf);
                map.push(mapping);
            }
            Some(map)
        } else {
            None
        };

        let indices = (0..data.len()).collect();
        Ok(Self {
            name,
            data,
            metric,
            is_expensive,
            indices,
            reordering,
        })
    }
}

impl<T: Send + Sync + Copy, U: Number> std::fmt::Debug for VecDataset<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Space")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<T: Send + Sync + Copy, U: Number> Dataset<T, U> for VecDataset<T, U> {
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

    fn get_reordered_index(&self, i: usize) -> Option<usize> {
        self.reordering.as_ref().map(|indices| indices[i])
    }

    fn one_to_one(&self, left: usize, right: usize) -> U {
        (self.metric)(self.data[left], self.data[right])
    }

    fn query_to_one(&self, query: T, index: usize) -> U {
        (self.metric)(query, self.data[index])
    }

    fn make_shards(self, max_cardinality: usize) -> Vec<Self> {
        let indices = {
            let mut indices = self.indices.clone();
            indices.shuffle(&mut rand::thread_rng());
            indices
        };

        indices
            .chunks(max_cardinality)
            .enumerate()
            .map(|(i, indices)| {
                let data = indices.iter().map(|&i| self.data[i]).collect::<Vec<_>>();
                let name = format!("{}-shard-{i}", self.name);
                Self::new(name, data, self.metric, self.is_expensive)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use distances::vectors::euclidean_sq;
    use symagen::random_data;
    use tempdir::TempDir;

    #[test]
    fn test_reordering_u32() {
        // 10 random 10 dimensional datasets reordered 10 times in 10 random ways
        let mut rng = rand::thread_rng();
        let name = "test".to_string();
        let cardinality = 10_000;

        for i in 0..10 {
            let dimensionality = 10;
            let reference_data = random_data::random_u32(cardinality, dimensionality, 0, 100_000, i);
            let reference_data = reference_data.iter().map(Vec::as_slice).collect::<Vec<_>>();
            for _ in 0..10 {
                let mut dataset = VecDataset::new(name.clone(), reference_data.clone(), euclidean_sq::<u32, u32>, false);
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
        let data: Vec<&[u32]> = data.iter().map(Vec::as_slice).collect();
        let permutation = vec![1, 3, 4, 0, 5, 2];

        let mut dataset = VecDataset::new("test".to_string(), data, euclidean_sq::<u32, u32>, false);

        dataset.reorder(&permutation);

        assert_eq!(
            dataset.data,
            vec![vec![4], vec![8], vec![10], vec![2], vec![12], vec![6],]
        );

        assert_eq!(dataset.get_reordered_index(0), Some(3));
        assert_eq!(
            dataset.get_reordered_index(0).map(|i| dataset.data[i]),
            Some([2].as_slice())
        );

        assert_eq!(dataset.get_reordered_index(1), Some(0));
        assert_eq!(
            dataset.get_reordered_index(1).map(|i| dataset.data[i]),
            Some([4].as_slice())
        );

        assert_eq!(dataset.get_reordered_index(2), Some(5));
        assert_eq!(
            dataset.get_reordered_index(2).map(|i| dataset.data[i]),
            Some([6].as_slice())
        );

        assert_eq!(dataset.get_reordered_index(3), Some(1));
        assert_eq!(
            dataset.get_reordered_index(3).map(|i| dataset.data[i]),
            Some([8].as_slice())
        );

        assert_eq!(dataset.get_reordered_index(4), Some(2));
        assert_eq!(
            dataset.get_reordered_index(4).map(|i| dataset.data[i]),
            Some([10].as_slice())
        );

        assert_eq!(dataset.get_reordered_index(5), Some(4));
        assert_eq!(
            dataset.get_reordered_index(5).map(|i| dataset.data[i]),
            Some([12].as_slice())
        );
    }

    #[test]
    fn test_save_load_deterministic() {
        let data = vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10]];
        let data: Vec<&[u32]> = data.iter().map(Vec::as_slice).collect();
        let tmp_dir = TempDir::new("save_load_deterministic").unwrap();
        let tmp_file = tmp_dir.path().join("datset.save");

        let mut dataset = VecDataset::new("test".to_string(), data, euclidean_sq::<u32, u32>, false);
        let indices = dataset.indices().iter().map(|x| *x).rev().collect::<Vec<usize>>();
        dataset.set_reordered_indices(&indices);
        dataset.save(&tmp_file).unwrap();

        let mut buffer = vec![];
        let other = VecDataset::<&[u32], u32>::load(&tmp_file, euclidean_sq, false, &mut buffer).unwrap();

        assert_eq!(other.data, dataset.data);
        assert_eq!(other.reordering, dataset.reordering);
        assert_eq!(dataset.cardinality(), other.cardinality());
    }

    #[test]
    fn test_save_load_random() {
        use rand::Rng;

        // Generate a random dataset. On even indices it generates a reordering map. Otherwise its not reordered
        let mut rng = rand::thread_rng();
        let tmp_dir = TempDir::new("save_load_deterministic").unwrap();
        for i in 0..5 {
            let (cardinality, dimensionality) = (rng.gen_range(1_000..5_0000), rng.gen_range(1..50));
            let reference_data = random_data::random_u32(cardinality, dimensionality, 0, 100_000, i);
            let reference_data = reference_data.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let tmp_file = tmp_dir.path().join(format!("datset_{}.save", i));

            let mut dataset = VecDataset::new("test".to_string(), reference_data, euclidean_sq::<u32, u32>, false);
            if i % 2 == 0 {
                let indices = dataset.indices().iter().map(|x| *x).rev().collect::<Vec<usize>>();
                dataset.set_reordered_indices(&indices);
            }
            dataset.save(&tmp_file).unwrap();

            let mut buffer = vec![];
            let other = VecDataset::<&[u32], u32>::load(&tmp_file, euclidean_sq, false, &mut buffer).unwrap();

            assert_eq!(other.data, dataset.data);
            assert_eq!(other.name, dataset.name);
            assert_eq!(other.reordering, dataset.reordering);
            assert_eq!(dataset.cardinality(), other.cardinality());
        }
    }
    #[test]
    fn test_unaligned_type_err() {
        let data = vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10]];
        let data: Vec<&[u32]> = data.iter().map(Vec::as_slice).collect();
        let tmp_dir = TempDir::new("save_load_deterministic").unwrap();
        let tmp_file = tmp_dir.path().join("datset.save");

        // Construct it with u32
        let mut dataset = VecDataset::new("test".to_string(), data, euclidean_sq::<u32, u32>, false);
        let indices = dataset.indices().iter().map(|x| *x).rev().collect::<Vec<usize>>();
        dataset.set_reordered_indices(&indices);
        dataset.save(&tmp_file).unwrap();

        let mut buffer = vec![];

        // Try to load it back in as f32
        let other = VecDataset::<&[f32], f32>::load(&tmp_file, euclidean_sq, false, &mut buffer);

        assert!(other.is_err());
    }
}
