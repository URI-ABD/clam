use super::_constructable::ConstructableNumber;
use super::reader::BatchedArrowReader;
use crate::Dataset;
use distances::Number;
use std::error::Error;

/// A dataset comprised of one or more batches of Arrow IPC files.
///
/// ## Note on IPC subset restriction
/// In lieu of writing an entire replacement IPC library, this functionality only supports
/// the small subset of the Arrow IPC format described below. Leveraging these restrictions,
/// we can achieve *significant* speedup and *significant* memory efficiency when reading
/// metadata and doing random access column reads.
///
/// At the moment, `BatchedArrowDataset` is only compatible with Arrow IPC datasets under
/// the following restrictions:
/// - Single, primitive type.
/// - Every batch has the same number of fields
/// - Single chunk per batch
///
/// Essentially, your datasets must be one statically sized type and in one chunk per batch.
/// If these requirements are met, your dataset should be readable (otherwise its a bug).
///
/// Another consequence of these assumptions is the that the type you give for `T` *must
/// match the corresponding type that actually exists in the file*. I.e. you must choose
/// `f32` if your dataset is `Float32`. Other choices will likely result in a crash from
/// file pointers being set incorrectly or just planinly incorrect results if the two
/// types are the same width.
///
/// ## Note on reordering
/// Importantly, `BatchedArrowDataset` acts as an interface to a set of files, so when reordering
/// one of these datasets, if you'd like persistent reordering between runs, use the
/// `BatchedArrowDataset::reorder_to_file` method which will not only reorder the dataset's indices
/// but also write out the reordering order to disk which will be read and used upon construction.
#[derive(Debug)]
pub struct BatchedArrowDataset<T: ConstructableNumber, U: Number> {
    name: String,
    metric: fn(&Vec<T>, &Vec<T>) -> U,
    metric_is_expensive: bool,
    reader: BatchedArrowReader<T>,
}

impl<T: ConstructableNumber, U: Number> BatchedArrowDataset<T, U> {
    /// Constructs a new `BatchedArrowDataset`
    ///
    /// The generic parameters `T` and `U` correspond to the type of each row in
    /// the dataset and the result of the distance measure respectively.
    ///
    /// # Args
    /// - `data_dir`: The directory where the batched Arrow IPC data is stored
    /// - `name`: The desired name of the dataset
    /// - `metric`: The desired distance metric
    /// - `metric_is_expensive`: True if and only if the distance measure is considred
    ///     expensive to compute.
    ///
    /// # Returns
    /// A result containing a constructed `BatchedArrowDataset`
    pub fn new(
        data_dir: &str,
        name: String,
        metric: fn(&Vec<T>, &Vec<T>) -> U,
        metric_is_expensive: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let reader = BatchedArrowReader::new(data_dir)?;
        Ok(Self {
            name,
            metric,
            metric_is_expensive,
            reader,
        })
    }

    pub fn get(&self, idx: usize) -> Vec<T> {
        // assert!(idx < self.cardinality(), "Index out of range");
        self.reader.get(idx)
    }

    /// Returns a row of the dataset at a given index
    ///
    /// # Notes
    /// This function will panic in the event of an invalid index (idx >= self.cardinality())
    ///
    /// # Args
    /// `idx`: The desired index
    ///
    /// # Returns
    /// The row at the provided index

    /// Performs a dataset reordering and then writes that reordering to disk in the
    /// dataset's `data_dir`.
    ///
    /// # Args
    /// - `indices`: The indices to reorder the dataset by
    pub fn reorder_to_file(&mut self, indices: &[usize]) -> Result<(), Box<dyn Error>> {
        self.reorder(indices);
        self.reader.write_reordering_map()?;

        Ok(())
    }

    /// Returns the reordered set of indices. This array is identical to `indices` if no
    /// reordering has taken place
    ///
    /// # Returns
    /// The reordered index array
    pub fn reordered_indices(&self) -> &[usize] {
        &self.reader.indices.reordered_indices
    }
}

impl<T: ConstructableNumber, U: Number> crate::Dataset<Vec<T>, U> for BatchedArrowDataset<T, U> {
    fn name(&self) -> &str {
        &self.name
    }

    // fn metric(&self) -> fn(&Vec<T>, &Vec<T>) -> U {
    //     self.metric
    // }

    fn cardinality(&self) -> usize {
        self.reader.indices.original_indices.len()
    }

    fn is_metric_expensive(&self) -> bool {
        self.metric_is_expensive
    }

    fn indices(&self) -> &[usize] {
        &self.reader.indices.original_indices
    }

    fn one_to_one(&self, left: usize, right: usize) -> U {
        (self.metric)(&self.get(left), &self.get(right))
    }

    fn query_to_one(&self, query: &Vec<T>, index: usize) -> U {
        (self.metric)(query, &self.get(index))
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.reader.indices.reordered_indices.swap(i, j);
    }

    fn set_reordered_indices(&mut self, indices: &[usize]) {
        self.reader.indices.reordered_indices = indices.to_vec();
    }

    fn get_reordered_index(&self, i: usize) -> Option<usize> {
        Some(self.reader.indices.reordered_indices[i])
    }
}
