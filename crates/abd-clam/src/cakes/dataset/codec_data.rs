//! An implementation of the Compression and Decompression traits.

use std::collections::HashMap;

use distances::Number;

use crate::{Dataset, Metric, MetricSpace};

use super::{Decodable, Decompressible};

/// A compressed dataset, that can be partially decompressed for search and
/// other applications.
///
/// A `CodecData` may only be built from a `Permutable` dataset, after the tree
/// has been built and the instances in the dataset have been permuted. This is
/// necessary for the `get` method to work correctly. Further, it is discouraged
/// to use the `get` method because it can be expensive if the instance being
/// retrieved is not the center of a cluster.
///
/// # Type Parameters
///
/// - `I`: The type of the instances in the dataset.
/// - `U`: The type of the numbers in the dataset.
/// - `M`: The type of the metadata associated with the instances.
pub struct CodecData<I, U, M> {
    /// The metric space of the dataset.
    metric: Metric<I, U>,
    /// The cardinality of the dataset.
    cardinality: usize,
    /// A hint for the dimensionality of the dataset.
    dimensionality_hint: (usize, Option<usize>),
    /// The metadata associated with the instances.
    metadata: Vec<M>,
    /// The centers of the clusters in the dataset.
    centers: HashMap<usize, I>,
    /// The bytes representing the leaf clusters as a flattened vector.
    leaf_bytes: Box<[u8]>,
    /// The offsets that indicate the start of the instances for each leaf
    /// cluster in the flattened vector.
    leaf_offsets: Vec<usize>,
}

impl<I, U, M> CodecData<I, U, M> {
    /// Returns the metadata associated with the instances in the dataset.
    #[must_use]
    pub fn metadata(&self) -> &[M] {
        &self.metadata
    }
}

impl<I: Decodable, U: Number, M> Decompressible<I, U> for CodecData<I, U, M> {}

impl<I: Decodable, U: Number, M> Dataset<I, U> for CodecData<I, U, M> {
    fn cardinality(&self) -> usize {
        self.cardinality
    }

    fn dimensionality_hint(&self) -> (usize, Option<usize>) {
        self.dimensionality_hint
    }

    fn get(&self, index: usize) -> &I {
        self.centers.get(&index).map_or_else(
            || {
                // Find the first offset that is larger than the given index.
                let offset_index = self
                    .leaf_offsets
                    .iter()
                    .position(|&offset| offset > index)
                    .unwrap_or(self.leaf_bytes.len());
                // The offset for the leaf is one less than that.
                let mut leaf_offset = self.leaf_offsets[offset_index - 1];
                // Decode the leaf.
                let _leaf = self.decode_leaf(&self.leaf_bytes, &mut leaf_offset, &self.centers);
                // Return the instance at the given index.
                // &leaf[index - leaf_offset]
                todo!()
            },
            |center| center,
        )
    }
}

impl<I, U: Number, M> MetricSpace<I, U> for CodecData<I, U, M> {
    fn identity(&self) -> bool {
        self.metric.identity()
    }

    fn non_negativity(&self) -> bool {
        self.metric.non_negativity()
    }

    fn symmetry(&self) -> bool {
        self.metric.symmetry()
    }

    fn triangle_inequality(&self) -> bool {
        self.metric.triangle_inequality()
    }

    fn expensive(&self) -> bool {
        self.metric.expensive()
    }

    fn distance_function(&self) -> fn(&I, &I) -> U {
        self.metric.distance_function()
    }
}
