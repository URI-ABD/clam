//! Provides the `Dataset` trait and the `Tabular` struct implementing it.

// TODO: Implement more structs for other types of datasets.
// For example:
// * FASTA/FASTQ files containing variable length genomic sequences.
// * Images. e.g. from SDSS-MaNGA dataset
// * Molecular graphs with Tanamoto distance.

use crate::prelude::*;

/// A `Dataset` represents a collection of instances. It provides access to
/// properties such as `cardinality` and `dimensionality` of the data. It lets
/// one estimate the memory consumption of loading large volumes of data into
/// memory. The main utility is that this trait provides access to individual
/// instances by index.
///
/// A `Dataset` and a `Metric` can be combined into a metric-`Space` for use in
/// CLAM.
pub trait Dataset<'a, T: Number>: std::fmt::Debug + Send + Sync {
    /// Ideally, the user will provide a different name for each dataset they
    /// initialize.
    fn name(&self) -> String;

    /// Returns the number of instances in the dataset.
    fn cardinality(&self) -> usize;

    /// Returns the dimensionality of the dataset
    fn dimensionality(&self) -> usize;

    fn indices(&self) -> Vec<usize>;

    /// Returns the instance at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - of the instance to return from the dataset.
    fn get(&self, index: usize) -> &'a [T];

    /// Returns a batch of indexed instances at once. The default implementation
    /// sequentially calls the `get` method. Users may have more efficient
    /// methods for this.
    ///
    /// # Arguments
    ///
    /// * `indices` - of instances to return from the dataset.
    fn get_batch(&self, indices: &[usize]) -> Vec<&'a [T]> {
        indices.iter().map(|&index| self.get(index)).collect()
    }

    /// Returns the size of the memory footprint of an instance in Bytes.
    fn max_instance_size(&self) -> usize {
        self.dimensionality() * (T::num_bytes() as usize)
    }

    /// Returns an upper bound on the memory footprint of the full dataset. The
    /// default implementation assumes that each instance in the dataset
    /// has the same memory footprint.
    fn approx_memory_size(&self) -> usize {
        self.cardinality() * self.max_instance_size()
    }

    /// Returns the batch-size to use given `available_memory`. This represents
    /// the largest number of instances that can fit in the given amount of
    /// memory.
    ///
    /// # Arguments
    ///
    /// * `available_memory` - in bytes
    fn max_batch_size(&self, available_memory: usize) -> usize {
        available_memory / self.max_instance_size()
    }
}

/// `Tabular` represents a dataset stored as a 2-dimensional array (represented)
/// as a nested Vec. Rows are instances and columns are features/attributes. The
/// data are kept in memory. When wrapped in a `TabularSpace`, this is
/// sufficient for most in-memory use-cases of CLAM.
///
/// This holds only a reference to the data and does not own it outright. This
/// might change in the future.
pub struct Tabular<'a, T: Number> {
    data: &'a [Vec<T>],
    name: String,
}

impl<'a, T: Number> std::fmt::Debug for Tabular<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Dataset")
            .field("name", &self.name)
            .field("cardinality", &self.cardinality())
            .field("dimensionality", &self.dimensionality())
            .finish()
    }
}

impl<'a, T: Number> Tabular<'a, T> {
    /// # Arguments
    ///
    /// `data` - Reference to the data to use.
    /// `name` - for the dataset. Ideally, this would be unique for each
    /// dataset.
    pub fn new(data: &'a [Vec<T>], name: String) -> Tabular<'a, T> {
        assert!(!data.is_empty());
        assert!(!data.first().unwrap().is_empty());
        Tabular { data, name }
    }
}

impl<'a, T: Number> Dataset<'a, T> for Tabular<'a, T> {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn cardinality(&self) -> usize {
        self.data.len()
    }

    fn dimensionality(&self) -> usize {
        self.data.first().unwrap().len()
    }

    fn indices(&self) -> Vec<usize> {
        (0..self.cardinality()).collect()
    }

    fn get(&self, index: usize) -> &'a [T] {
        &self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::Dataset;
    use super::Tabular;

    #[test]
    fn test_dataset() {
        let data = vec![vec![1., 2., 3.], vec![3., 3., 1.]];
        let dataset = Tabular::new(&data, "test_dataset".to_string());

        assert_eq!(dataset.cardinality(), 2);
        assert_eq!(dataset.get(0), data[0]);
    }
}
