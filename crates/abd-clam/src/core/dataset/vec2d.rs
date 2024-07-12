//! A dataset of a Vec of instances.

use core::{fmt::Debug, ops::Index};

use std::{
    fs::File,
    io::{BufWriter, Read, Write},
    path::Path,
};

use distances::Number;
use rayon::prelude::*;

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
/// - `M`: The type of the metadata associated with each instance.
#[derive(Debug, Clone)]
pub struct VecDataset<I: Instance, U: Number, M: Instance> {
    /// The name of the dataset.
    pub(crate) name: String,
    /// The data of the dataset.
    pub(crate) data: Vec<I>,
    /// The metric of the dataset.
    pub(crate) metric: fn(&I, &I) -> U,
    /// Whether the metric is expensive to compute.
    pub(crate) is_expensive: bool,
    /// The reordering of the dataset after building the tree.
    pub(crate) permuted_indices: Option<Vec<usize>>,
    /// Metadata about the dataset.
    pub(crate) metadata: Vec<M>,
}

impl<I: Instance, U: Number> VecDataset<I, U, usize> {
    /// Creates a new dataset.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the dataset.
    /// * `data`: The vector of instances.
    /// * `metric`: The metric for computing distances between instances.
    /// * `is_expensive`: Whether the metric is expensive to compute.
    pub fn new(name: String, data: Vec<I>, metric: fn(&I, &I) -> U, is_expensive: bool) -> Self {
        let metadata = (0..data.len()).collect();
        Self {
            name,
            data,
            metric,
            is_expensive,
            permuted_indices: None,
            metadata,
        }
    }
}

impl<I: Instance, U: Number, M: Instance> VecDataset<I, U, M> {
    /// Assigns metadata to the dataset.
    ///
    /// # Arguments
    ///
    /// * `metadata`: The metadata to assign to the dataset.
    ///
    /// # Returns
    ///
    /// The dataset with the metadata assigned.
    ///
    /// # Errors
    ///
    /// * If the metadata is not the same length as the dataset.
    pub fn assign_metadata<Mn: Instance>(self, metadata: Vec<Mn>) -> Result<VecDataset<I, U, Mn>, String> {
        if metadata.len() == self.data.len() {
            // If there is a permutation, permute the metadata as well.
            let metadata = if let Some(permutation) = self.permuted_indices.as_ref() {
                permutation.par_iter().map(|&index| metadata[index].clone()).collect()
            } else {
                metadata
            };

            Ok(VecDataset {
                name: self.name,
                data: self.data,
                metric: self.metric,
                is_expensive: self.is_expensive,
                permuted_indices: self.permuted_indices,
                metadata,
            })
        } else {
            Err(format!(
                "Invalid metadata. Expected metadata of length {}, got metadata of length {}",
                self.cardinality(),
                metadata.len()
            ))
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

    /// A reference to the underlying metadata.
    #[must_use]
    pub fn metadata(&self) -> &[M] {
        &self.metadata
    }

    /// Moves the underlying metadata out of the dataset.
    #[must_use]
    pub fn metadata_owned(self) -> Vec<M> {
        self.metadata
    }

    /// A reference to the metadata of a specific instance.
    #[must_use]
    pub fn metadata_of(&self, index: usize) -> &M {
        &self.metadata[index]
    }
}

impl<I: Instance, U: Number, M: Instance> Index<usize> for VecDataset<I, U, M> {
    type Output = I;

    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index)
    }
}

impl<I: Instance, U: Number, M: Instance> Dataset<I, U> for VecDataset<I, U, M> {
    fn clone_with_new_metric(&self, metric: fn(&I, &I) -> U, is_expensive: bool, name: String) -> Self {
        Self {
            name,
            data: self.data.clone(),
            metric,
            is_expensive,
            permuted_indices: self.permuted_indices.clone(),
            metadata: self.metadata.clone(),
        }
    }

    fn type_name() -> String {
        format!("VecDataset<{}, {}, {}>", I::type_name(), U::type_name(), M::type_name())
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

    fn set_permuted_indices(&mut self, indices: Option<&[usize]>) {
        self.permuted_indices = indices.map(<[usize]>::to_vec);
    }

    fn swap(&mut self, left: usize, right: usize) -> Result<(), String> {
        self.data.swap(left, right);
        self.metadata.swap(left, right);
        Ok(())
    }

    fn permuted_indices(&self) -> Option<&[usize]> {
        self.permuted_indices.as_deref()
    }

    fn permute_instances(&mut self, permutation: &[usize]) -> Result<(), String> {
        if permutation.len() != self.data.len() {
            return Err(format!(
                "Invalid permutation. Expected permutation of length {}, got permutation of length {}",
                self.cardinality(),
                permutation.len()
            ));
        }

        self.data = permutation.par_iter().map(|&index| self.data[index].clone()).collect();
        self.metadata = permutation
            .par_iter()
            .map(|&index| self.metadata[index].clone())
            .collect();

        self.set_permuted_indices(Some(permutation));

        Ok(())
    }

    fn make_shards(mut self, max_cardinality: usize) -> Vec<Self> {
        let mut shards = Vec::new();
        let mut metadata = self.metadata.clone();

        while self.data.len() > max_cardinality {
            // Create a new name for the shard.
            let name = format!("{}-shard-{}", self.name, shards.len());

            // Split the data.
            let at = self.data.len() - max_cardinality;
            let data = self.data.split_off(at);

            // Create the shard, assign the metadata, and add it to the list of shards.
            shards.push(
                VecDataset::new(name, data, self.metric, self.is_expensive)
                    .assign_metadata(metadata.split_off(at))
                    .unwrap_or_else(|_| unreachable!("We just split this dataset at the same indices.")),
            );
        }

        self.name = format!("{}-shard-{}", self.name, shards.len());
        shards.push(self);

        shards
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        let mut handle = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);

        // Write header (Basic protection against reading bad data)
        let type_name = Self::type_name();
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
        let cardinality_bytes = self.data.len().to_le_bytes();
        handle.write_all(&cardinality_bytes).map_err(|e| e.to_string())?;

        // If the dataset was permuted, write the permutation map.
        let permutation = self
            .permuted_indices
            .as_ref()
            .map_or(Vec::new(), |p| p.iter().flat_map(|i| i.to_le_bytes()).collect());
        let permutation_bytes = permutation.len().to_le_bytes();
        handle
            .write_all(&permutation_bytes)
            .and_then(|()| handle.write_all(&permutation))
            .map_err(|e| e.to_string())?;

        // Write individual vectors
        for row in &self.data {
            row.save(&mut handle)?;
        }

        // Write number of metadata
        handle.write_all(&cardinality_bytes).map_err(|e| e.to_string())?;

        // Write metadata
        for meta in &self.metadata {
            meta.save(&mut handle)?;
        }

        Ok(())
    }

    fn load(path: &Path, metric: fn(&I, &I) -> U, is_expensive: bool) -> Result<Self, String> {
        let mut handle = File::open(path).map_err(|e| e.to_string())?;

        // Check that the type name matches.
        {
            // Read the number of bytes in the type name
            let mut num_type_bytes = vec![0; usize::num_bytes()];
            handle.read_exact(&mut num_type_bytes).map_err(|e| e.to_string())?;
            let num_type_bytes = <usize as Number>::from_le_bytes(&num_type_bytes);

            // Read the type name
            let mut type_buf = vec![0; num_type_bytes];
            handle.read_exact(&mut type_buf).map_err(|e| e.to_string())?;
            let type_name = String::from_utf8(type_buf).map_err(|e| e.to_string())?;

            // Check that the type name matches.
            let actual_type_name = Self::type_name();
            if type_name != actual_type_name {
                return Err(format!(
                    "Invalid type. File has data of type {type_name} but dataset was constructed with type {actual_type_name}"
                ));
            }
        };

        // Read the given name of the dataset
        let name = {
            let mut num_name_bytes = vec![0; usize::num_bytes()];
            handle.read_exact(&mut num_name_bytes).map_err(|e| e.to_string())?;
            let num_name_bytes = <usize as Number>::from_le_bytes(&num_name_bytes);

            // Get the dataset's name
            let mut name_buf = vec![0; num_name_bytes];
            handle.read_exact(&mut name_buf).map_err(|e| e.to_string())?;
            String::from_utf8(name_buf).map_err(|e| e.to_string())?
        };

        // Read the cardinality
        let cardinality = {
            let mut cardinality_buf = vec![0; usize::num_bytes()];
            handle.read_exact(&mut cardinality_buf).map_err(|e| e.to_string())?;
            <usize as Number>::from_le_bytes(&cardinality_buf)
        };

        // Read the permutation, if it exists
        let permutation = {
            let mut permutation_buf = vec![0; usize::num_bytes()];
            handle.read_exact(&mut permutation_buf).map_err(|e| e.to_string())?;
            if <usize as Number>::from_le_bytes(&permutation_buf) == 0 {
                None
            } else {
                let mut permutation_buf = vec![0; 8 * cardinality];
                handle.read_exact(&mut permutation_buf).map_err(|e| e.to_string())?;
                let permutation = permutation_buf
                    .chunks(8)
                    .map(<usize as Number>::from_le_bytes)
                    .collect::<Vec<_>>();
                Some(permutation)
            }
        };

        // Read the individual vectors
        let data = (0..cardinality)
            .map(|_| I::load(&mut handle))
            .collect::<Result<Vec<_>, _>>()?;

        // Read the number of metadata
        let num_metadata = {
            let mut num_metadata_buf = vec![0; usize::num_bytes()];
            handle.read_exact(&mut num_metadata_buf).map_err(|e| e.to_string())?;
            <usize as Number>::from_le_bytes(&num_metadata_buf)
        };

        let metadata = (0..num_metadata)
            .map(|_| M::load(&mut handle))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            name,
            data,
            metric,
            is_expensive,
            permuted_indices: permutation,
            metadata,
        })
    }
}
