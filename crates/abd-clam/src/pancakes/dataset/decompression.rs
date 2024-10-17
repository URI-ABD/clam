//! Traits and an implementation for decompressing datasets.

use std::collections::HashMap;

use crate::{dataset::ParDataset, Dataset};

use super::Encodable;

/// For items that can be decoded from a byte array or in terms of a reference.
///
/// We provide a blanket implementation for all types that implement `Number`.
pub trait Decodable: Encodable {
    /// Decodes the value from a byte array.
    fn from_bytes(bytes: &[u8]) -> Self;

    /// Decodes the value in terms of a reference.
    fn decode(reference: &Self, bytes: &[u8]) -> Self;
}

/// Given `Decodable` items, a compressed dataset can be decompressed.
pub trait Decompressible<I: Decodable>: Dataset<I> {
    /// Returns the centers of the clusters in the tree associated with this
    /// dataset.
    fn centers(&self) -> &HashMap<usize, I>;

    /// Returns the bytes slices representing all leaves' offsets and compressed
    /// bytes.
    fn leaf_bytes(&self) -> &[(usize, Box<[u8]>)];

    /// Decodes all the items of a leaf cluster in terms of its center.
    fn decode_leaf(&self, bytes: &[u8]) -> Vec<I> {
        let mut items = Vec::new();

        let mut offset = 0;
        let arg_center = crate::utils::read_number::<usize>(bytes, &mut offset);
        let center = &self.centers()[&arg_center];

        let cardinality = crate::utils::read_number::<usize>(bytes, &mut offset);

        for _ in 0..cardinality {
            let encoding = crate::utils::read_encoding(bytes, &mut offset);
            let item = I::decode(center, &encoding);
            items.push(item);
        }

        items
    }
}

/// Parallel version of [`Decompressible`](crate::pancakes::dataset::decompression::Decompressible).
pub trait ParDecompressible<I: Decodable + Send + Sync>: Decompressible<I> + ParDataset<I> {
    /// Parallel version of [`Decompressible::decode_leaf`](crate::pancakes::dataset::decompression::Decompressible::decode_leaf).
    fn par_decode_leaf(&self, bytes: &[u8]) -> Vec<I> {
        self.decode_leaf(bytes)
    }
}

impl<T: distances::Number> Decodable for T {
    fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes)
    }

    fn decode(_: &Self, bytes: &[u8]) -> Self {
        Self::from_bytes(bytes)
    }
}
