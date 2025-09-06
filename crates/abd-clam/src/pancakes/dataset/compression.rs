//! Traits that define the behavior of compression algorithms.

use num::traits::ToBytes;
use rayon::prelude::*;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, Dataset, DistanceValue, FlatVec};

/// Something that can encode items into a byte array or in terms of a reference.
pub trait Encoder<I> {
    /// Encodes the item into a byte array.
    fn to_byte_array(&self, item: &I) -> Box<[u8]>;

    /// Encodes the item in terms of a reference.
    fn encode(&self, item: &I, reference: &I) -> Box<[u8]>;
}

/// Parallel version of [`Encoder`](crate::pancakes::dataset::compression::Encoder).
///
/// The default implementation of `ParEncoder` simply delegates to the
/// non-parallel version.
pub trait ParEncoder<I: Send + Sync>: Encoder<I> + Send + Sync {
    /// Parallel version of [`Encoder::to_byte_array`](crate::pancakes::dataset::compression::Encoder::to_byte_array).
    fn par_to_byte_array(&self, item: &I) -> Box<[u8]> {
        self.to_byte_array(item)
    }

    /// Parallel version of [`Encoder::encode`](crate::pancakes::dataset::compression::Encoder::encode).
    fn par_encode(&self, item: &I, reference: &I) -> Box<[u8]> {
        self.encode(item, reference)
    }
}

/// Given `Encodable` items, a dataset can be compressed.
pub trait Compressible<I, Enc: Encoder<I>>: Dataset<I> {
    /// Encodes all the items of leaf clusters in terms of their centers.
    ///
    /// # Returns
    ///
    /// - A vector of byte arrays, each containing the encoded items of a leaf cluster.
    fn encode_leaves<'a, T: DistanceValue + 'a, C: Cluster<T>>(
        &self,
        root: &'a C,
        encoder: &Enc,
    ) -> Vec<(&'a C, Box<[u8]>)> {
        root.leaves()
            .into_iter()
            .map(|leaf| {
                let center = self.get(leaf.arg_center());
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&leaf.arg_center().to_le_bytes());
                bytes.extend_from_slice(&leaf.cardinality().to_le_bytes());
                for i in leaf.indices() {
                    let encoding = encoder.encode(self.get(i), center);
                    bytes.extend_from_slice(&encoding.len().to_le_bytes());
                    bytes.extend_from_slice(&encoding);
                }
                (leaf, bytes.into_boxed_slice())
            })
            .collect()
    }
}

/// Parallel version of [`Compressible`](crate::pancakes::dataset::compression::Compressible).
pub trait ParCompressible<I: Send + Sync, Enc: ParEncoder<I>>: Compressible<I, Enc> + ParDataset<I> {
    /// Parallel version of [`Compressible::encode_leaves`](crate::pancakes::dataset::compression::Compressible::encode_leaves).
    fn par_encode_leaves<'a, T: DistanceValue + Send + Sync + 'a, C: ParCluster<T>>(
        &self,
        root: &'a C,
        encoder: &Enc,
    ) -> Vec<(&'a C, Box<[u8]>)> {
        root.leaves()
            .into_par_iter()
            .map(|leaf| {
                let center = self.get(leaf.arg_center());
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&leaf.arg_center().to_le_bytes());
                bytes.extend_from_slice(&leaf.cardinality().to_le_bytes());
                for i in leaf.indices() {
                    let encoding = encoder.par_encode(self.get(i), center);
                    bytes.extend_from_slice(&encoding.len().to_le_bytes());
                    bytes.extend_from_slice(&encoding);
                }
                (leaf, bytes.into_boxed_slice())
            })
            .collect()
    }
}

impl<T: DistanceValue + ToBytes<Bytes = Vec<u8>>> Encoder<T> for T {
    fn to_byte_array(&self, item: &T) -> Box<[u8]> {
        item.to_le_bytes().into_boxed_slice()
    }

    fn encode(&self, item: &T, _: &T) -> Box<[u8]> {
        item.to_le_bytes().into_boxed_slice()
    }
}

impl<T: DistanceValue + ToBytes<Bytes = Vec<u8>> + Send + Sync> ParEncoder<T> for T {}

impl<I, Me, Enc: Encoder<I>> Compressible<I, Enc> for FlatVec<I, Me> {}

impl<I: Send + Sync, Me: Send + Sync, Enc: ParEncoder<I>> ParCompressible<I, Enc> for FlatVec<I, Me> {}
