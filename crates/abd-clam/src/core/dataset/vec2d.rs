//! A dataset of a Vec of instances.

use core::{ops::Index, slice::SliceIndex};
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use distances::Number;

use crate::Dataset;

use super::Instance;

/// A `Dataset` of a `Vec` of instances.
///
/// This may be used for any data that can fit in memory. It is not recommended for large datasets.
///
/// # Type Parameters
///
/// - `T`: The type of the instances in the `Dataset`.
/// - `U`: The type of the distance values between instances.
pub struct VecDataset<I: Instance, U: Number> {
    /// The name of the dataset.
    name: String,
    /// The data of the dataset.
    data: Vec<I>,
    /// The metric of the dataset.
    metric: fn(&I, &I) -> U,
    /// Whether the metric is expensive to compute.
    is_expensive: bool,
    /// The reordering of the dataset after building the tree.
    permuted_indices: Option<Vec<usize>>,
}

impl<I: Instance, U: Number> VecDataset<I, U> {
    /// Creates a new dataset.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the dataset.
    /// * `data`: The vector of instances.
    /// * `metric`: The metric for computing distances between instances.
    /// * `is_expensive`: Whether the metric is expensive to compute.
    pub fn new(name: String, data: Vec<I>, metric: fn(&I, &I) -> U, is_expensive: bool) -> Self {
        Self {
            name,
            data,
            metric,
            is_expensive,
            permuted_indices: None,
        }
    }

    /// A reference to the underlying data.
    #[must_use]
    pub fn data(&self) -> &[I] {
        &self.data
    }

    /// Moves the underlying data out of the dataset.
    #[must_use]
    pub fn data_owned(self) -> Vec<I> {
        self.data
    }
}

impl<I: Instance, U: Number, Idx> Index<Idx> for VecDataset<I, U>
where
    Idx: SliceIndex<[I], Output = I>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        self.data.index(index)
    }
}

impl<I: Instance, U: Number> Dataset<I, U> for VecDataset<I, U> {
    fn type_name(&self) -> String {
        format!("VecDataset<{}>", I::type_name())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn cardinality(&self) -> usize {
        self.data.len()
    }

    fn is_metric_expensive(&self) -> bool {
        self.is_expensive
    }

    fn metric(&self) -> fn(&I, &I) -> U {
        self.metric
    }

    fn permute_instances(&mut self, permutation: &[usize]) -> Result<(), String> {
        let n = permutation.len();

        // The "source index" represents the index that we hope to swap to
        let mut source_index: usize;

        // INVARIANT: After each iteration of the loop, the elements of the
        // sub-array [0..i] are in the correct position.
        for i in 0..n - 1 {
            source_index = permutation[i];

            // If the element at is already at the correct position, we can
            // just skip.
            if source_index != i {
                // Here we're essentially following the cycle. We *know* by
                // the invariant that all elements to the left of i are in
                // the correct position, so what we're doing is following
                // the cycle until we find an index to the right of i. Which,
                // because we followed the position changes, is the correct
                // index to swap.
                while source_index < i {
                    source_index = permutation[source_index];
                }

                // We swap to the correct index. Importantly, this index is always
                // to the right of i, we do not modify any index to the left of i.
                // Thus, because we followed the cycle to the correct index to swap,
                // we know that the element at i, after this swap, is in the correct
                // position.
                self.data.swap(source_index, i);
            }
        }

        // Inverse mapping
        self.permuted_indices = Some(permutation.to_vec());

        Ok(())
    }

    fn permuted_indices(&self) -> Option<&[usize]> {
        self.permuted_indices.as_deref()
    }

    fn make_shards(mut self, max_cardinality: usize) -> Vec<Self> {
        let mut shards = Vec::new();

        while self.data.len() > max_cardinality {
            let at = self.data.len() - max_cardinality;
            let chunk = self.data.split_off(at);

            let name = format!("{}-shard-{}", self.name, shards.len());
            shards.push(Self::new(name, chunk, self.metric, self.is_expensive));
        }
        self.name = format!("{}-shard-{}", self.name, shards.len());
        shards.push(self);

        shards
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        let mut handle = File::create(path).map_err(|e| e.to_string())?;

        // Write header (Basic protection against reading bad data)
        let type_name = self.type_name();
        handle
            .write_all(&type_name.len().to_le_bytes())
            .and_then(|()| handle.write_all(type_name.as_bytes()))
            .map_err(|e| e.to_string())?;

        // Write dataset name
        let name = self.name.clone();
        handle
            .write_all(&name.len().to_le_bytes())
            .and_then(|()| handle.write_all(name.as_bytes()))
            .map_err(|e| e.to_string())?;

        // Write cardinality
        let cardinality = self.data.len();
        let cardinality_bytes = cardinality.to_le_bytes();
        handle.write_all(&cardinality_bytes).map_err(|e| e.to_string())?;

        // Write individual vectors
        for row in &self.data {
            row.save(&mut handle)?;
        }

        // If the dataset was permuted, write the permutation map.
        let permutation = self.permuted_indices.as_ref().map_or(Vec::new(), |permutation| {
            permutation.iter().flat_map(|i| i.to_le_bytes()).collect()
        });
        let permutation_bytes = permutation.len().to_le_bytes();
        handle
            .write_all(&permutation_bytes)
            .and_then(|()| handle.write_all(&permutation))
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    fn load(path: &Path, metric: fn(&I, &I) -> U, is_expensive: bool) -> Result<Self, String> {
        let mut handle = File::open(path).map_err(|e| e.to_string())?;

        // Read the number of bytes in the type name
        let mut num_type_bytes = vec![0; usize::num_bytes()];
        handle.read_exact(&mut num_type_bytes).map_err(|e| e.to_string())?;
        let num_type_bytes = <usize as Number>::from_le_bytes(&num_type_bytes);

        // Read the type name
        let mut type_buf = vec![0; num_type_bytes];
        handle.read_exact(&mut type_buf).map_err(|e| e.to_string())?;
        let type_name = String::from_utf8(type_buf).map_err(|e| e.to_string())?;

        // If the type name doesn't match, return an error
        let actual_type_name = format!("VecDataset<{}>", I::type_name());
        if type_name != actual_type_name {
            return Err(format!(
                "Invalid type. File has data of type {type_name} but dataset was constructed with type {actual_type_name}"
            ));
        }

        // Read the number of bytes in the dataset's name
        let mut num_name_bytes = vec![0; usize::num_bytes()];
        handle.read_exact(&mut num_name_bytes).map_err(|e| e.to_string())?;
        let num_name_bytes = <usize as Number>::from_le_bytes(&num_name_bytes);

        // Get the dataset's name
        let mut name_buf = vec![0; num_name_bytes];
        handle.read_exact(&mut name_buf).map_err(|e| e.to_string())?;
        let name = String::from_utf8(name_buf).map_err(|e| e.to_string())?;

        // Read in the cardinality
        let mut cardinality_buf = vec![0; usize::num_bytes()];
        handle.read_exact(&mut cardinality_buf).map_err(|e| e.to_string())?;
        let cardinality = <usize as Number>::from_le_bytes(&cardinality_buf);

        // Read in the individual vectors
        let mut data = Vec::with_capacity(cardinality);
        for _ in 0..cardinality {
            data.push(I::load(&mut handle)?);
        }

        // Check if there's a permutation
        let mut permutation_buf = vec![0; usize::num_bytes()];
        handle.read_exact(&mut permutation_buf).map_err(|e| e.to_string())?;
        let permutation = if <usize as Number>::from_le_bytes(&permutation_buf) == 0 {
            None
        } else {
            let mut permutation_buf = vec![0; 8 * cardinality];
            handle.read_exact(&mut permutation_buf).map_err(|e| e.to_string())?;
            let permutation = permutation_buf
                .chunks(8)
                .map(<usize as Number>::from_le_bytes)
                .collect::<Vec<_>>();
            Some(permutation)
        };

        Ok(Self {
            name,
            data,
            metric,
            is_expensive,
            permuted_indices: permutation,
        })
    }
}
