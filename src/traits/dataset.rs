//! `Dataset` trait and some structs implementing it.
//!
//! Contains the declaration and definition of the `Dataset` trait and the
//! `Tabular` struct implementing Dataset to serves most of the use cases for `CLAM`.

// TODO: Implement more structs for other types of datasets.
// For example:
// * FASTA/FASTQ files containing variable length genomic sequences.
// * Images. e.g. from SDSS-MaNGA dataset
// * Molecular graphs with Tanamoto distance.

use crate::prelude::*;

/// All datasets supplied to `CLAM` must implement this trait.
pub trait Dataset<T: Number>: std::fmt::Debug + Send + Sync {
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
    /// * `index` - The index of the instance to return from the dataset.
    ///
    fn get(&self, index: usize) -> Vec<T>;

    /// Returns the size of the memory footprint of an instance in Bytes.
    fn max_instance_size(&self) -> usize {
        self.dimensionality() * (T::num_bytes() as usize)
    }

    fn approx_memory_size(&self) -> usize {
        self.cardinality() * self.max_instance_size()
    }

    /// Returns the batch-size to use given `available_memory`.
    ///
    /// # Arguments
    ///
    /// * `available_memory` - in bytes
    fn max_batch_size(&self, available_memory: usize) -> usize {
        available_memory / self.max_instance_size()
    }
}

/// RowMajor represents a dataset stored as a 2-dimensional array
/// where rows are instances and columns are features/attributes.
///
/// A wrapper around an `ndarray::Array2` of data, along with a `Metric`,
/// to provide an interface for computing distances between instances
/// contained within the dataset.
///
/// The resulting structure can make use of caching techniques to prevent
/// repeated (potentially expensive) calls to its internal distance function.
pub struct Tabular<'a, T: Number> {
    /// 2D array of data
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
    pub fn new(data: &'a [Vec<T>], name: String) -> Tabular<'a, T> {
        assert!(!data.is_empty());
        assert!(!data.first().unwrap().is_empty());
        Tabular { data, name }
    }

    //noinspection RsSelfConvention
    pub fn as_arc_dataset(&self) -> &dyn Dataset<T> {
        self
    }
}

impl<'a, T: Number> Dataset<T> for Tabular<'a, T> {
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

    fn get(&self, index: usize) -> Vec<T> {
        self.data[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::Dataset;
    use super::Tabular;

    #[test]
    fn test_dataset() {
        let data = vec![vec![1., 2., 3.], vec![3., 3., 1.]];
        let row_0 = vec![1., 2., 3.];
        let dataset = Tabular::new(&data, "test_dataset".to_string());

        assert_eq!(dataset.cardinality(), 2);
        assert_eq!(dataset.get(0), row_0);
    }
}
