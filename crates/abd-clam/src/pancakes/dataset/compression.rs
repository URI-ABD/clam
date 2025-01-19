//! Traits that define the behavior of compression algorithms.

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

/// For items can be encoded into a byte array or in terms of a reference.
///
/// We provide a blanket implementation for all types that implement `Number`.
pub trait Encodable {
    /// Converts the value to a byte array.
    fn as_bytes(&self) -> Box<[u8]>;

    /// Encodes the value in terms of a reference.
    fn encode(&self, reference: &Self) -> Box<[u8]>;
}

/// Given `Encodable` items, a dataset can be compressed.
pub trait Compressible<I: Encodable>: Dataset<I> {
    /// Encodes all the items of leaf clusters in terms of their centers.
    ///
    /// # Returns
    ///
    /// - A vector of byte arrays, each containing the encoded items of a leaf cluster.
    fn encode_leaves<'a, T: Number + 'a, C: Cluster<T>>(&self, root: &'a C) -> Vec<(&'a C, Box<[u8]>)> {
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

/// Parallel version of [`Compressible`](crate::pancakes::dataset::compression::Compressible).
pub trait ParCompressible<I: Encodable + Send + Sync>: Compressible<I> + ParDataset<I> {
    /// Parallel version of [`Compressible::encode_leaves`](crate::pancakes::dataset::compression::Compressible::encode_leaves).
    fn par_encode_leaves<'a, T: Number + 'a, C: ParCluster<T>>(&self, root: &'a C) -> Vec<(&'a C, Box<[u8]>)> {
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

impl<T: Number> Encodable for T {
    fn as_bytes(&self) -> Box<[u8]> {
        self.to_le_bytes().into_boxed_slice()
    }

    fn encode(&self, _: &Self) -> Box<[u8]> {
        self.as_bytes()
    }
}
