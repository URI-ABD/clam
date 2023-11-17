//! A dataset of a Vec of instances.

use core::{fmt::Debug, ops::Index};

use std::{
    fs::File,
    io::{BufWriter, Read, Write},
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
#[derive(Debug)]
pub struct VecDataset<I: Instance, U: Number, M: Instance> {
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
    /// Metadata about the dataset.
    metadata: Option<Vec<M>>,
}

impl<I: Instance, U: Number, M: Instance> VecDataset<I, U, M> {
    /// Creates a new dataset.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the dataset.
    /// * `data`: The vector of instances.
    /// * `metric`: The metric for computing distances between instances.
    /// * `is_expensive`: Whether the metric is expensive to compute.
    pub fn new(
        name: String,
        data: Vec<I>,
        metric: fn(&I, &I) -> U,
        is_expensive: bool,
        metadata: Option<Vec<M>>,
    ) -> Self {
        Self {
            name,
            data,
            metric,
            is_expensive,
            permuted_indices: None,
            metadata,
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
    pub fn metadata(&self) -> Option<&[M]> {
        self.metadata.as_deref()
    }

    /// Moves the underlying metadata out of the dataset.
    #[must_use]
    pub fn metadata_owned(self) -> Option<Vec<M>> {
        self.metadata
    }

    /// A reference to the metadata of a specific instance.
    #[must_use]
    pub fn metadata_of(&self, index: usize) -> Option<&M> {
        self.metadata().map(|m| &m[index])
    }
}

impl<I: Instance, U: Number, M: Instance> Index<usize> for VecDataset<I, U, M> {
    type Output = I;

    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index)
    }
}

impl<I: Instance, U: Number, M: Instance> Dataset<I, U> for VecDataset<I, U, M> {
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

    fn set_permuted_indices(&mut self, indices: Option<&[usize]>) {
        self.permuted_indices = indices.map(<[usize]>::to_vec);
    }

    fn swap(&mut self, left: usize, right: usize) -> Result<(), String> {
        self.data.swap(left, right);
        if let Some(v) = self.metadata.as_mut() {
            v.swap(left, right);
        };
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
            let meta_chunk = self.metadata.as_mut().map(|m| m.split_off(at));

            let name = format!("{}-shard-{}", self.name, shards.len());
            shards.push(Self::new(name, chunk, self.metric, self.is_expensive, meta_chunk));
        }
        self.name = format!("{}-shard-{}", self.name, shards.len());
        shards.push(self);

        shards
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        let mut handle = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);

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
        let num_metadata = self.metadata.as_ref().map_or(0, Vec::len).to_le_bytes();
        handle.write_all(&num_metadata).map_err(|e| e.to_string())?;

        // Write metadata
        if let Some(metadata) = &self.metadata {
            for meta in metadata {
                meta.save(&mut handle)?;
            }
        }

        Ok(())
    }

    fn load(path: &Path, metric: fn(&I, &I) -> U, is_expensive: bool) -> Result<Self, String> {
        let mut handle = File::open(path).map_err(|e| e.to_string())?;

        // Check the type name.
        {
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
        };

        // Read the name
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

        let metadata = if num_metadata == 0 {
            None
        } else {
            let metadata = (0..num_metadata)
                .map(|_| M::load(&mut handle))
                .collect::<Result<Vec<_>, _>>()?;
            Some(metadata)
        };

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
