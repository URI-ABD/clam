//! A dataset for searching in a compressed space.

use std::collections::{HashMap, HashSet};

use distances::{number::UInt, Number};

use crate::{Cluster, Dataset, Instance};

use super::{DecoderFn, EncoderFn, SquishyBall};

/// A `Dataset` that allows for searching in a compressed space.
#[derive(Debug)]
pub struct CodecData<I: Instance, U: UInt, M: Instance> {
    /// The root of the squishy ball tree.
    root: SquishyBall<U>,
    /// The subset of the dataset that contains the centers of the clusters in the tree.
    centers: HashMap<usize, I>,
    /// The compressed data for the squished clusters.
    leaf_data: LeafData<I>,
    /// The distance function.
    metric: fn(&I, &I) -> U,
    /// Whether the distance function is expensive to compute.
    is_expensive: bool,
    /// Metadata for the dataset.
    metadata: Vec<M>,
    /// The reordering of the dataset after building the tree.
    permuted_indices: Vec<usize>,
}

impl<I: Instance, U: UInt, M: Instance> CodecData<I, U, M> {
    /// Creates a new `CodecData`.
    ///
    /// # Arguments
    ///
    /// * `root`: The root of the squishy ball tree.
    /// * `data`: The dataset to compress.
    /// * `encoder`: The encoding function.
    /// * `decoder`: The decoding function.
    /// * `metadata`: Metadata for the dataset.
    ///
    /// # Returns
    ///
    /// The `CodecData` for the dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if any leaf data could not be encoded.
    pub fn new<D: Dataset<I, U>>(
        mut root: SquishyBall<U>,
        data: &D,
        encoder: EncoderFn<I>,
        decoder: DecoderFn<I>,
        metadata: Vec<M>,
    ) -> Result<Self, String> {
        let permuted_indices = data
            .permuted_indices()
            .map_or_else(|| (0..data.cardinality()).collect(), <[usize]>::to_vec);

        // Apply the criteria to the tree
        root.apply_criteria();

        // Build the centers
        let leaves = root.compressible_leaves_mut();
        let centers = leaves.iter().map(|c| c.arg_center()).collect::<HashSet<_>>();
        let centers = centers
            .into_iter()
            .map(|i| (i, data[i].clone()))
            .collect::<HashMap<_, _>>();

        // Build the leaves' data
        let mut bytes = Vec::new();
        for leaf in leaves.into_iter().filter(|c| c.squish()) {
            leaf.set_codec_offset(bytes.len());
            let center = &data[leaf.arg_center()];
            let encodings = leaf
                .indices()
                .map(|i| encoder(center, &data[i]))
                .collect::<Result<Vec<_>, _>>()?;
            let num_encodings = encodings.len();
            bytes.extend_from_slice(&num_encodings.to_le_bytes());
            for encoding in encodings {
                let len = encoding.len();
                bytes.extend_from_slice(&len.to_le_bytes());
                bytes.extend_from_slice(&encoding);
            }
        }

        let bytes = bytes.into_boxed_slice();
        let leaf_data = LeafData { bytes, decoder };

        Ok(Self {
            root,
            centers,
            leaf_data,
            metric: data.metric(),
            is_expensive: data.is_metric_expensive(),
            metadata,
            permuted_indices,
        })
    }

    /// Loads the data for a leaf.
    ///
    /// # Arguments
    ///
    /// * `leaf`: The leaf to load.
    ///
    /// # Returns
    ///
    /// The data for the leaf.
    ///
    /// # Errors
    ///
    /// Returns an error if any leaf data could not be decoded.
    pub fn load_leaf_data(&self, leaf: &SquishyBall<U>) -> Result<Vec<I>, String> {
        let offset = leaf.codec_offset().ok_or("Leaf has no codec offset")?;
        let center = &self.centers[&leaf.arg_center()];
        self.leaf_data.load_leaf(center, offset)
    }

    /// Returns the root of the squishy ball tree.
    pub const fn root(&self) -> &SquishyBall<U> {
        &self.root
    }

    /// Returns the centers of the clusters in the tree.
    pub const fn centers(&self) -> &HashMap<usize, I> {
        &self.centers
    }

    /// Returns the metadata for the dataset.
    pub fn metadata(&self) -> &[M] {
        &self.metadata
    }

    /// Returns the reordering of the dataset after building the tree.
    pub fn permuted_indices(&self) -> &[usize] {
        &self.permuted_indices
    }

    /// Returns the distance function.
    pub fn metric(&self) -> fn(&I, &I) -> U {
        self.metric
    }

    /// Returns whether the distance function is expensive to compute.
    pub const fn is_expensive(&self) -> bool {
        self.is_expensive
    }
}

/// The compressed data for the squished clusters.
#[derive(Debug)]
struct LeafData<I: Instance> {
    /// The compressed data for the squished clusters.
    pub bytes: Box<[u8]>,
    /// The decoding function.
    pub decoder: DecoderFn<I>,
}

impl<I: Instance> LeafData<I> {
    /// Loads the data for a leaf.
    ///
    /// # Arguments
    ///
    /// * `center`: The center of the leaf.
    /// * `offset`: The offset in the compressed data where the leaf data starts.
    ///
    /// # Returns
    ///
    /// The data for the leaf.
    ///
    /// # Errors
    ///
    /// Returns an error if any leaf data could not be decoded.
    fn load_leaf(&self, center: &I, offset: usize) -> Result<Vec<I>, String> {
        // Read the number of encodings.
        let num_encodings = {
            let bytes = &self.bytes[offset..(offset + usize::num_bytes())];
            <usize as Number>::from_le_bytes(bytes)
        };

        let mut data = Vec::with_capacity(num_encodings);
        let mut offset = offset + usize::num_bytes();
        for _ in 0..num_encodings {
            // Read the length of the encoding.
            let len = {
                let bytes = &self.bytes[offset..(offset + usize::num_bytes())];
                <usize as Number>::from_le_bytes(bytes)
            };
            offset += usize::num_bytes();

            // Read the encoding.
            let encoding = {
                let bytes = &self.bytes[offset..(offset + len)];
                offset += len;
                (self.decoder)(center, bytes)?
            };

            data.push(encoding);
        }

        Ok(data)
    }
}
