//! Traits that define the behavior of compression algorithms.

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

/// A trait that defines how a value can be encoded in terms of a reference.
pub trait Encodable: Clone {
    /// Converts the value to a byte array.
    fn as_bytes(&self) -> Box<[u8]>;

    /// Encodes the value in terms of a reference.
    fn encode(&self, reference: &Self) -> Box<[u8]>;
}

/// A trait that defines how a dataset can be compressed.
pub trait Compressible<I: Encodable, U: Number>: Dataset<I, U> + Sized {
    /// Encodes all the instances of leaf clusters in terms of their centers.
    ///
    /// # Returns
    ///
    /// - A vector of byte arrays, each containing the encoded instances of a leaf cluster.
    fn encode_leaves<'a, D: Dataset<I, U>, C: Cluster<I, U, D>>(&self, root: &'a C) -> Vec<(&'a C, Box<[u8]>)>
    where
        U: 'a,
    {
        root.leaves()
            .into_iter()
            .map(|leaf| {
                let center = self.get(leaf.arg_center());
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&leaf.arg_center().to_le_bytes());
                bytes.extend_from_slice(&leaf.cardinality().to_le_bytes());
                for i in leaf.indices() {
                    let encoding = self.get(i).encode(center);
                    bytes.extend_from_slice(&encoding.len().to_le_bytes());
                    bytes.extend_from_slice(&encoding);
                }
                (leaf, bytes.into_boxed_slice())
            })
            .collect()
    }
}

/// A trait that defines how a dataset can be compressed.
pub trait ParCompressible<I: Encodable + Send + Sync, U: Number>: Compressible<I, U> + ParDataset<I, U> + Sized {
    /// Encodes all the instances of leaf clusters in terms of their centers.
    ///
    /// # Returns
    ///
    /// - A flattened vector of encoded instances.
    /// - A vector of offsets that indicate the start of the instances for each
    ///   leaf cluster in the flattened vector.
    /// - A vector of cumulative cardinalities of leaves.
    fn par_encode_leaves<'a, D: ParDataset<I, U>, C: ParCluster<I, U, D>>(&self, root: &'a C) -> Vec<(&'a C, Box<[u8]>)>
    where
        U: 'a,
    {
        root.leaves()
            .into_par_iter()
            .map(|leaf| {
                let center = self.get(leaf.arg_center());
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&leaf.arg_center().to_le_bytes());
                bytes.extend_from_slice(&leaf.cardinality().to_le_bytes());
                for i in leaf.indices() {
                    let encoding = self.get(i).encode(center);
                    bytes.extend_from_slice(&encoding.len().to_le_bytes());
                    bytes.extend_from_slice(&encoding);
                }
                (leaf, bytes.into_boxed_slice())
            })
            .collect()
    }
}
