//! A `FlatVec` is a dataset that is stored as a flat vector.

use distances::Number;
use serde::{Deserialize, Serialize};

use super::{metric_space::ParMetricSpace, Dataset, Metric, MetricSpace, ParDataset, Permutable};

/// A `FlatVec` is a dataset that is stored as a flat vector.
///
/// The instances are stored as a flat vector.
///
/// # Type Parameters
///
/// - `I`: The type of the instances in the dataset.
/// - `U`: The type of the distance values.
/// - `M`: The type of the metadata associated with the instances.
#[derive(Clone, Serialize, Deserialize)]
pub struct FlatVec<I, U, M> {
    /// The metric space of the dataset.
    #[serde(skip)]
    pub(crate) metric: Metric<I, U>,
    /// The instances in the dataset.
    pub(crate) instances: Vec<I>,
    /// A hint for the dimensionality of the dataset.
    pub(crate) dimensionality_hint: (usize, Option<usize>),
    /// The permutation of the instances.
    pub(crate) permutation: Vec<usize>,
    /// The metadata associated with the instances.
    pub(crate) metadata: Vec<M>,
    /// The name of the dataset.
    pub(crate) name: String,
}

impl<I, U> FlatVec<I, U, usize> {
    /// Creates a new `FlatVec`.
    ///
    /// # Parameters
    ///
    /// - `instances`: The instances in the dataset.
    /// - `metric`: The metric space of the dataset.
    ///
    /// # Returns
    ///
    /// A new `FlatVec`.
    ///
    /// # Errors
    ///
    /// * If the instances are empty.
    pub fn new(instances: Vec<I>, metric: Metric<I, U>) -> Result<Self, String> {
        if instances.is_empty() {
            Err("The instances are empty.".to_string())
        } else {
            let permutation = (0..instances.len()).collect::<Vec<_>>();
            let metadata = permutation.clone();
            Ok(Self {
                metric,
                instances,
                dimensionality_hint: (0, None),
                permutation,
                metadata,
                name: "Unknown FlatVec".to_string(),
            })
        }
    }
}

impl<T, U> FlatVec<Vec<T>, U, usize> {
    /// Creates a new `FlatVec` from tabular data.
    ///
    /// The data are assumed to be a 2d array where each row is an instance.
    /// The dimensionality of the dataset is set to the number of columns in the data.
    ///
    /// # Parameters
    ///
    /// - `instances`: The instances in the dataset.
    /// - `metric`: The metric space of the dataset.
    ///
    /// # Returns
    ///
    /// A new `FlatVec`.
    ///
    /// # Errors
    ///
    /// * If the instances are empty.
    pub fn new_array(instances: Vec<Vec<T>>, metric: Metric<Vec<T>, U>) -> Result<Self, String> {
        if instances.is_empty() {
            Err("The instances are empty.".to_string())
        } else {
            let dimensionality = instances[0].len();
            let permutation = (0..instances.len()).collect::<Vec<_>>();
            let metadata = permutation.clone();
            Ok(Self {
                metric,
                instances,
                dimensionality_hint: (dimensionality, Some(dimensionality)),
                permutation,
                metadata,
                name: "Unknown FlatVec".to_string(),
            })
        }
    }
}

impl<I, U, M> FlatVec<I, U, M> {
    /// Deconstructs the `FlatVec` into its members.
    ///
    /// # Returns
    ///
    /// - The `Metric` of the dataset.
    /// - The instances in the dataset.
    /// - A hint for the dimensionality of the dataset.
    /// - The permutation of the instances.
    /// - The metadata associated with the instances.
    #[allow(clippy::type_complexity)]
    #[must_use]
    pub fn deconstruct(self) -> (Metric<I, U>, Vec<I>, (usize, Option<usize>), Vec<usize>, Vec<M>, String) {
        (
            self.metric,
            self.instances,
            self.dimensionality_hint,
            self.permutation,
            self.metadata,
            self.name,
        )
    }

    /// Sets a lower bound for the dimensionality of the dataset.
    #[must_use]
    pub const fn with_dim_lower_bound(mut self, lower_bound: usize) -> Self {
        self.dimensionality_hint.0 = lower_bound;
        self
    }

    /// Sets an upper bound for the dimensionality of the dataset.
    #[must_use]
    pub const fn with_dim_upper_bound(mut self, upper_bound: usize) -> Self {
        self.dimensionality_hint.1 = Some(upper_bound);
        self
    }

    /// Returns the metadata associated with the instances.
    #[must_use]
    pub fn metadata(&self) -> &[M] {
        &self.metadata
    }

    /// Assigns metadata to the instances.
    ///
    /// # Parameters
    ///
    /// - `metadata`: The metadata to assign to the instances.
    ///
    /// # Returns
    ///
    /// The dataset with the metadata assigned to the instances.
    ///
    /// # Errors
    ///
    /// * If the metadata length does not match the number of instances.
    pub fn with_metadata<Me>(self, mut metadata: Vec<Me>) -> Result<FlatVec<I, U, Me>, String> {
        if metadata.len() == self.instances.len() {
            metadata.permute(&self.permutation);
            Ok(FlatVec {
                metric: self.metric,
                instances: self.instances,
                dimensionality_hint: self.dimensionality_hint,
                permutation: self.permutation,
                metadata,
                name: self.name,
            })
        } else {
            Err(format!(
                "The metadata length does not match the number of instances. {} vs {}",
                metadata.len(),
                self.instances.len()
            ))
        }
    }

    /// Get the instances in the dataset.
    #[must_use]
    pub fn instances(&self) -> &[I] {
        &self.instances
    }
}

impl<I, U: Number, M> MetricSpace<I, U> for FlatVec<I, U, M> {
    fn metric(&self) -> &Metric<I, U> {
        &self.metric
    }

    fn set_metric(&mut self, metric: Metric<I, U>) {
        self.metric = metric;
    }
}

impl<I: Send + Sync, U: Number, M: Send + Sync> ParMetricSpace<I, U> for FlatVec<I, U, M> {}

impl<I, U: Number, M> Dataset<I, U> for FlatVec<I, U, M> {
    fn name(&self) -> &str {
        &self.name
    }

    /// Changes the name of the dataset.
    fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    fn cardinality(&self) -> usize {
        self.instances.len()
    }

    fn dimensionality_hint(&self) -> (usize, Option<usize>) {
        self.dimensionality_hint
    }

    fn get(&self, index: usize) -> &I {
        &self.instances[index]
    }
}

impl<I: Send + Sync, U: Number, M: Send + Sync> ParDataset<I, U> for FlatVec<I, U, M> {}

impl<I, U: Number, M> Permutable for FlatVec<I, U, M> {
    fn permutation(&self) -> Vec<usize> {
        self.permutation.clone()
    }

    fn set_permutation(&mut self, permutation: &[usize]) {
        self.permutation = permutation.to_vec();
    }

    fn swap_two(&mut self, i: usize, j: usize) {
        self.instances.swap(i, j);
        self.permutation.swap(i, j);
        self.metadata.swap(i, j);
    }
}

#[cfg(feature = "ndarray-bindings")]
impl<T: ndarray_npy::ReadableElement + Copy, U> FlatVec<Vec<T>, U, usize> {
    /// Reads a `VecDataset` from a `.npy` file.
    ///
    /// # Parameters
    ///
    /// - `path`: The path to the `.npy` file.
    /// - `metric`: The metric space of the dataset.
    /// - `name`: The name of the dataset. If `None`, the name of the file is used.
    ///
    /// # Errors
    ///
    /// * If the path is invalid.
    /// * If the file cannot be read.
    /// * If the instances cannot be converted to a `Vec`.
    pub fn read_npy<P: AsRef<std::path::Path>>(path: P, metric: Metric<Vec<T>, U>) -> Result<Self, String> {
        let arr: ndarray::Array2<T> = ndarray_npy::read_npy(path).map_err(|e| e.to_string())?;
        let instances = arr.axis_iter(ndarray::Axis(0)).map(|row| row.to_vec()).collect();
        Self::new_array(instances, metric)
    }
}

#[cfg(feature = "ndarray-bindings")]
impl<T: ndarray_npy::WritableElement + Copy, U> FlatVec<Vec<T>, U, usize> {
    /// Writes the `VecDataset` to a `.npy` file in the given directory.
    ///
    /// # Parameters
    ///
    /// - `dir`: The directory in which to write the `.npy` file.
    /// - `name`: The name of the file. If `None`, the name of the dataset is used.
    ///
    /// # Returns
    ///
    /// The path to the written file.
    ///
    /// # Errors
    ///
    /// * If the path is invalid.
    /// * If the file cannot be created.
    /// * If the instances cannot be converted to an `Array2`.
    /// * If the `Array2` cannot be written.
    pub fn write_npy<P: AsRef<std::path::Path>>(&self, dir: P, name: &str) -> Result<std::path::PathBuf, String> {
        let path = dir.as_ref().join(name);
        let shape = (self.instances.len(), self.dimensionality_hint.0);
        let v = self.instances.iter().flat_map(|row| row.iter().copied()).collect();
        let arr: ndarray::Array2<T> = ndarray::Array2::from_shape_vec(shape, v).map_err(|e| e.to_string())?;
        ndarray_npy::write_npy(&path, &arr).map_err(|e| e.to_string())?;
        Ok(path)
    }
}

#[cfg(feature = "csv")]
impl<T: std::str::FromStr + Copy, U> FlatVec<Vec<T>, U, usize> {
    /// Reads a `VecDataset` from a `.csv` file.
    ///
    /// # Parameters
    ///
    /// - `path`: The path to the `.csv` file.
    /// - `metric`: The metric space of the dataset.
    /// - `delimiter`: The delimiter used in the `.csv` file.
    /// - `has_headers`: Whether to treat the first row as headers.
    ///
    /// # Errors
    ///
    /// * If the path is invalid.
    /// * If the file cannot be read.
    /// * If the types in the file are not convertible to `T` using string parsing.
    /// * If the instances cannot be converted to a `Vec`.
    pub fn read_csv<P: AsRef<std::path::Path>>(path: P, metric: Metric<Vec<T>, U>) -> Result<Self, String> {
        let mut reader = csv::ReaderBuilder::new().from_path(path).map_err(|e| e.to_string())?;
        let instances = reader
            .records()
            .map(|record| {
                record.map_err(|e| e.to_string()).and_then(|record| {
                    record
                        .iter()
                        .map(|field| {
                            field
                                .parse::<T>()
                                .map_err(|_| "Could not parse field in csv".to_string())
                        })
                        .collect()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new_array(instances, metric)
    }
}

#[cfg(feature = "csv")]
impl<T: std::string::ToString + Copy, U, M> FlatVec<Vec<T>, U, M> {
    /// Writes the `VecDataset` to a `.csv` file with the given path.
    ///
    /// # Parameters
    ///
    /// - `path`: The path to the `.csv` file.
    /// - `delimiter`: The delimiter to use in the `.csv` file.
    ///
    /// # Errors
    ///
    /// * If the path is invalid.
    /// * If the file cannot be created.
    pub fn to_csv<P: AsRef<std::path::Path>>(&self, path: P, delimiter: u8) -> Result<(), String> {
        let mut writer = csv::WriterBuilder::new()
            .delimiter(delimiter)
            .from_path(path)
            .map_err(|e| e.to_string())?;
        for instance in &self.instances {
            writer
                .write_record(instance.iter().map(T::to_string))
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// Tests for the `FlatVec` struct.
#[cfg(test)]
mod tests {
    use crate::{dataset::ParDataset, Dataset, Metric, MetricSpace, Permutable};

    use super::FlatVec;

    #[test]
    fn creation() -> Result<(), String> {
        let instances = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let distance_function = |a: &Vec<i32>, b: &Vec<i32>| distances::vectors::manhattan(a, b);
        let metric = Metric::new(distance_function, false);

        let dataset = FlatVec::new(instances.clone(), metric.clone())?;
        assert_eq!(dataset.cardinality(), 3);
        assert_eq!(dataset.dimensionality_hint(), (0, None));

        let dataset = FlatVec::new_array(instances, metric)?;
        assert_eq!(dataset.cardinality(), 3);
        assert_eq!(dataset.dimensionality_hint(), (2, Some(2)));

        Ok(())
    }

    #[cfg(feature = "ndarray-bindings")]
    #[test]
    fn npy_io() -> Result<(), String> {
        use std::ffi::OsStr;

        let instances = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let distance_function = |a: &Vec<i32>, b: &Vec<i32>| distances::vectors::manhattan(a, b);
        let metric = Metric::new(distance_function, false);
        let dataset = FlatVec::new_array(instances, metric.clone())?;

        let tmp_dir = tempdir::TempDir::new("testing").map_err(|e| e.to_string())?;
        let path = dataset.write_npy(&tmp_dir, "test.npy")?;

        let new_dataset = FlatVec::read_npy(&path, metric.clone())?;
        assert_eq!(new_dataset.cardinality(), 3);
        assert_eq!(new_dataset.dimensionality_hint(), (2, Some(2)));
        for i in 0..dataset.cardinality() {
            assert_eq!(dataset.get(i), new_dataset.get(i));
        }

        let path = dataset.write_npy(&tmp_dir, "test-test.npy")?;
        assert_eq!(path.file_name().map(OsStr::to_str), Some(Some("test-test.npy")));

        let new_dataset = FlatVec::read_npy(&path, metric.clone())?;
        assert_eq!(new_dataset.cardinality(), 3);
        assert_eq!(new_dataset.dimensionality_hint(), (2, Some(2)));
        for i in 0..dataset.cardinality() {
            assert_eq!(dataset.get(i), new_dataset.get(i));
        }

        let new_dataset = FlatVec::read_npy(&path, metric)?;
        assert_eq!(new_dataset.cardinality(), 3);
        assert_eq!(new_dataset.dimensionality_hint(), (2, Some(2)));
        for i in 0..dataset.cardinality() {
            assert_eq!(dataset.get(i), new_dataset.get(i));
        }

        Ok(())
    }

    #[test]
    fn metricity() -> Result<(), String> {
        let instances = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let distance_function = |a: &Vec<i32>, b: &Vec<i32>| distances::vectors::manhattan(a, b);
        let metric = Metric::new(distance_function, false);
        let dataset = FlatVec::new_array(instances, metric)?;

        assert_eq!(dataset.get(0), &vec![1, 2]);
        assert_eq!(dataset.get(1), &vec![3, 4]);
        assert_eq!(dataset.get(2), &vec![5, 6]);

        assert_eq!(Dataset::one_to_one(&dataset, 0, 1), 4);
        assert_eq!(Dataset::one_to_one(&dataset, 1, 2), 4);
        assert_eq!(Dataset::one_to_one(&dataset, 2, 0), 8);
        assert_eq!(
            Dataset::one_to_many(&dataset, 0, &[0, 1, 2]),
            vec![(0, 0), (1, 4), (2, 8)]
        );
        assert_eq!(
            Dataset::many_to_many(&dataset, &[0, 1], &[1, 2]),
            vec![vec![(0, 1, 4), (0, 2, 8)], vec![(1, 1, 0), (1, 2, 4)]]
        );

        assert_eq!(dataset.query_to_one(&vec![0, 0], 0), 3);
        assert_eq!(dataset.query_to_one(&vec![0, 0], 1), 7);
        assert_eq!(dataset.query_to_one(&vec![0, 0], 2), 11);
        assert_eq!(
            dataset.query_to_many(&vec![0, 0], &[0, 1, 2]),
            vec![(0, 3), (1, 7), (2, 11)]
        );

        Ok(())
    }

    #[test]
    fn permutations() -> Result<(), String> {
        struct SwapTracker {
            data: FlatVec<Vec<i32>, i32, usize>,
            count: usize,
        }

        impl MetricSpace<Vec<i32>, i32> for SwapTracker {
            fn metric(&self) -> &Metric<Vec<i32>, i32> {
                &self.data.metric
            }

            fn set_metric(&mut self, metric: Metric<Vec<i32>, i32>) {
                self.data.metric = metric;
            }
        }

        impl Dataset<Vec<i32>, i32> for SwapTracker {
            fn name(&self) -> &str {
                self.data.name()
            }

            fn with_name(mut self, name: &str) -> Self {
                self.data = self.data.with_name(name);
                self
            }

            fn cardinality(&self) -> usize {
                self.data.cardinality()
            }

            fn dimensionality_hint(&self) -> (usize, Option<usize>) {
                self.data.dimensionality_hint()
            }

            fn get(&self, index: usize) -> &Vec<i32> {
                self.data.get(index)
            }
        }

        impl Permutable for SwapTracker {
            fn permutation(&self) -> Vec<usize> {
                self.data.permutation()
            }

            fn set_permutation(&mut self, permutation: &[usize]) {
                self.data.set_permutation(permutation);
            }

            fn swap_two(&mut self, i: usize, j: usize) {
                self.data.swap_two(i, j);
                self.count += 1;
            }
        }

        let instances = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6],
            vec![7, 8],
            vec![9, 10],
            vec![11, 12],
        ];
        let distance_function = |a: &Vec<i32>, b: &Vec<i32>| distances::vectors::manhattan(a, b);
        let metric = Metric::new(distance_function, false);
        let data = FlatVec::new_array(instances.clone(), metric.clone())?;
        let mut swap_tracker = SwapTracker { data, count: 0 };

        swap_tracker.swap_two(0, 2);
        assert_eq!(swap_tracker.permutation(), &[2, 1, 0, 3, 4, 5]);
        assert_eq!(swap_tracker.count, 1);
        for (i, &j) in swap_tracker.permutation().iter().enumerate() {
            assert_eq!(swap_tracker.get(i), &instances[j]);
        }

        swap_tracker.swap_two(0, 4);
        assert_eq!(swap_tracker.permutation(), &[4, 1, 0, 3, 2, 5]);
        assert_eq!(swap_tracker.count, 2);
        for (i, &j) in swap_tracker.permutation().iter().enumerate() {
            assert_eq!(swap_tracker.get(i), &instances[j]);
        }

        let data = FlatVec::new_array(instances.clone(), metric)?;
        let mut data = SwapTracker { data, count: 0 };
        let permutation = vec![2, 1, 0, 5, 4, 3];
        data.permute(&permutation);
        assert_eq!(data.permutation(), permutation);
        assert_eq!(data.count, 2);
        for (i, &j) in data.permutation().iter().enumerate() {
            assert_eq!(data.get(i), &instances[j]);
        }

        Ok(())
    }

    #[test]
    fn linear_search() -> Result<(), String> {
        let instances = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6],
            vec![7, 8],
            vec![9, 10],
            vec![11, 12],
        ];
        let distance_function = |a: &Vec<i32>, b: &Vec<i32>| distances::vectors::manhattan(a, b);
        let metric = Metric::new(distance_function, false);

        let dataset = FlatVec::new_array(instances, metric)?;
        let query = vec![3, 3]; // distances: [3, 1, 5, 9, 13, 17]

        let mut result = dataset.knn(&query, 2);
        result.sort_unstable_by_key(|x| x.0);
        assert_eq!(result, vec![(0, 3), (1, 1)]);

        let mut result = dataset.rnn(&query, 5);
        result.sort_unstable_by_key(|x| x.0);
        assert_eq!(result, vec![(0, 3), (1, 1), (2, 5)]);

        let mut result = dataset.par_knn(&query, 2);
        result.sort_unstable_by_key(|x| x.0);
        assert_eq!(result, vec![(0, 3), (1, 1)]);

        let mut result = dataset.par_rnn(&query, 5);
        result.sort_unstable_by_key(|x| x.0);
        assert_eq!(result, vec![(0, 3), (1, 1), (2, 5)]);

        Ok(())
    }

    #[test]
    fn ser_de() -> Result<(), String> {
        type Fv = FlatVec<Vec<i32>, i32, usize>;

        let instances = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let distance_function = |a: &Vec<i32>, b: &Vec<i32>| distances::vectors::manhattan(a, b);
        let metric = Metric::new(distance_function, false);
        let dataset: Fv = FlatVec::new_array(instances, metric.clone())?;

        let serialized: Vec<u8> = bincode::serialize(&dataset).map_err(|e| e.to_string())?;
        let mut deserialized: Fv = bincode::deserialize(&serialized).map_err(|e| e.to_string())?;
        deserialized.set_metric(metric);

        assert_eq!(dataset.cardinality(), deserialized.cardinality());
        assert_eq!(dataset.dimensionality_hint(), deserialized.dimensionality_hint());
        assert_eq!(dataset.permutation(), deserialized.permutation());
        assert_eq!(dataset.metadata(), deserialized.metadata());
        for i in 0..dataset.cardinality() {
            assert_eq!(dataset.get(i), deserialized.get(i));
        }

        Ok(())
    }
}
