//! A `Dataset` from a numpy memory-mapped array.

use std::path::{Path, PathBuf};

use distances::Number;
use ndarray::prelude::*;
use ndarray_npy::{ViewElement, ViewMutElement, ViewMutNpyExt, ViewNpyExt};

use crate::Dataset;

/// A `Dataset` from a numpy memory-mapped array.
#[derive(Debug)]
pub struct NpyMmap<'a, T: Number + ViewElement + ViewMutElement + ViewMutNpyExt<'a>, U: Number> {
    /// The path to the dataset.
    pub(crate) path: PathBuf,
    /// The name of the dataset.
    pub(crate) name: &'a str,
    /// The memory-mapped array handle.
    reader: memmap2::MmapMut,
    /// The metric to use.
    pub(crate) metric: fn(ArrayView1<T>, ArrayView1<T>) -> U,
    /// Whether the metric is expensive to compute.
    pub(crate) is_expensive: bool,
    /// The indices of the dataset.
    pub(crate) indices: Vec<usize>,
    /// The reordering of the indices.
    pub(crate) reordering: Option<Vec<usize>>,
}

impl<'a, T: Number + ViewElement + ViewMutElement + ViewMutNpyExt<'a>, U: Number> NpyMmap<'a, T, U> {
    /// Creates a new `NpyMmap` dataset.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory where the dataset is located.
    /// * `name` - The name of the npy file.
    /// * `metric` - The metric to use.
    /// * `is_expensive` - Whether the metric is expensive to compute.
    #[allow(dead_code)]
    pub fn new(
        directory: &Path,
        name: &'a str,
        metric: fn(ArrayView1<T>, ArrayView1<T>) -> U,
        is_expensive: bool,
    ) -> Result<Self, String> {
        let path = directory.join(name);
        let file = std::fs::File::open(&path)
            .map_err(|error| format!("Error: Failed to open your dataset at {path:?}. {error:}"))?;
        let reader = unsafe {
            memmap2::MmapMut::map_mut(&file)
                .map_err(|error| format!("Error: Failed to memory-map your dataset at {path:?}. {error:}"))?
        };
        let data = ArrayView2::<T>::view_npy(&reader)
            .map_err(|error| format!("Error: Failed to read your dataset at {path:?}. {error:}"))?;
        let indices = (0..data.nrows()).collect();

        Ok(Self {
            path,
            name,
            reader,
            metric,
            is_expensive,
            indices,
            reordering: None,
        })
    }

    /// Reads the dataset from the internal handle and returns a read-only view.
    fn view(&self) -> ArrayView2<T> {
        ArrayView2::<T>::view_npy(&self.reader)
            .map_err(|error| format!("Error: Failed to read your dataset at {:?}. {:}", self.path, error))
            .unwrap_or_else(|reason| unreachable!("File was previously read successfully: {reason}"))
    }

    /// Reads the dataset from the internal handle and returns a mutable view.
    fn view_mut(&mut self) -> ArrayViewMut2<T> {
        ArrayViewMut2::<T>::view_mut_npy(&mut self.reader)
            .map_err(|error| format!("Error: Failed to read your dataset at {:?}. {:}", self.path, error))
            .unwrap_or_else(|reason| unreachable!("File was previously read successfully: {reason}"))
    }
}

impl<'a, T: Number + ViewElement + ViewMutElement + ViewMutNpyExt<'a>, U: Number> Dataset<ArrayView1<'a, T>, U>
    for NpyMmap<'a, T, U>
{
    fn name(&self) -> &str {
        self.name
    }

    fn cardinality(&self) -> usize {
        self.indices.len()
    }

    fn is_metric_expensive(&self) -> bool {
        self.is_expensive
    }

    fn indices(&self) -> &[usize] {
        &self.indices
    }

    fn metric(&self) -> fn(ArrayView1<T>, ArrayView1<T>) -> U {
        self.metric
    }

    fn swap(&mut self, i: usize, j: usize) {
        let mut data = self.view_mut();

        let tmp = data.row(i).to_owned();
        let y = data.row(j).to_owned();

        let mut x = data.row_mut(i);
        x.assign(&y);

        let mut y = data.row_mut(j);
        y.assign(&tmp);
    }

    fn set_reordered_indices(&mut self, indices: &[usize]) {
        self.reordering = Some(indices.iter().map(|&i| indices[i]).collect());
    }

    fn get_reordered_index(&self, i: usize) -> Option<usize> {
        self.reordering.as_ref().map(|indices| indices[i])
    }

    fn one_to_one(&self, left: usize, right: usize) -> U {
        let data = self.view();
        (self.metric)(data.row(left), data.row(right))
    }

    fn one_to_many(&self, left: usize, right: &[usize]) -> Vec<U> {
        let data = self.view();
        right
            .iter()
            .map(|&i| (self.metric)(data.row(left), data.row(i)))
            .collect()
    }

    fn many_to_many(&self, left: &[usize], right: &[usize]) -> Vec<Vec<U>> {
        let data = self.view();
        left.iter()
            .map(|&i| right.iter().map(|&j| (self.metric)(data.row(i), data.row(j))).collect())
            .collect()
    }

    fn query_to_one(&self, query: ArrayView1<T>, index: usize) -> U {
        let data = self.view();
        (self.metric)(query, data.row(index))
    }

    fn query_to_many(&self, query: ArrayView1<T>, indices: &[usize]) -> Vec<U> {
        let data = self.view();
        indices.iter().map(|&i| (self.metric)(query, data.row(i))).collect()
    }
}
