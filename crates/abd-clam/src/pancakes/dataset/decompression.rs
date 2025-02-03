//! Traits and an implementation for decompressing datasets.

use std::collections::HashMap;

use distances::Number;

use crate::{dataset::ParDataset, Dataset};

/// Something that can decode items from a byte array or in terms of a reference.
pub trait Decoder<I> {
    /// Decodes the item from a byte array.
    #[allow(clippy::wrong_self_convention)]
    fn from_byte_array(&self, bytes: &[u8]) -> I;

    /// Decodes the item in terms of a reference.
    fn decode(&self, bytes: &[u8], reference: &I) -> I;
}

/// Parallel version of [`Decoder`](crate::pancakes::dataset::compression::Decoder).
///
/// The default implementation of `ParDecoder` simply delegates to the
/// non-parallel version.
pub trait ParDecoder<I: Send + Sync>: Decoder<I> + Send + Sync {
    /// Parallel version of [`Decoder::from_byte_array`](crate::pancakes::dataset::compression::Decoder::from_byte_array).
    fn par_from_byte_array(&self, bytes: &[u8]) -> I {
        self.from_byte_array(bytes)
    }

    /// Parallel version of [`Decoder::decode`](crate::pancakes::dataset::compression::Decoder::decode).
    fn par_decode(&self, bytes: &[u8], reference: &I) -> I {
        self.decode(bytes, reference)
    }
}

/// Given `Decodable` items, a compressed dataset can be decompressed.
pub trait Decompressible<I, Dec: Decoder<I>>: Dataset<I> {
    /// Returns the centers of the clusters in the tree associated with this
    /// dataset.
    fn centers(&self) -> &HashMap<usize, I>;

    /// Returns the bytes slices representing all leaves' offsets and compressed
    /// bytes.
    fn leaf_bytes(&self) -> &[(usize, Box<[u8]>)];

    /// Decodes all the items of a leaf cluster in terms of its center.
    fn decode_leaf(&self, bytes: &[u8], decoder: &Dec) -> Vec<(usize, I)> {
        let mut items = Vec::new();

        let mut offset = 0;
        let arg_center = crate::utils::read_number::<usize>(bytes, &mut offset);
        let center = &self.centers()[&arg_center];

        let cardinality = crate::utils::read_number::<usize>(bytes, &mut offset);

        for i in 0..cardinality {
            let encoding = crate::utils::read_encoding(bytes, &mut offset);
            let item = decoder.decode(&encoding, center);
            items.push((offset + i, item));
        }

        items
    }
}

/// Parallel version of [`Decompressible`](crate::pancakes::dataset::decompression::Decompressible).
pub trait ParDecompressible<I: Send + Sync, Dec: ParDecoder<I>>: Decompressible<I, Dec> + ParDataset<I> {
    /// Parallel version of [`Decompressible::decode_leaf`](crate::pancakes::dataset::decompression::Decompressible::decode_leaf).
    fn par_decode_leaf(&self, bytes: &[u8], decoder: &Dec) -> Vec<(usize, I)> {
        self.decode_leaf(bytes, decoder)
    }
}

impl<T: Number> Decoder<T> for T {
    fn from_byte_array(&self, bytes: &[u8]) -> T {
        T::from_le_bytes(bytes)
    }

    fn decode(&self, bytes: &[u8], _: &T) -> T {
        T::from_le_bytes(bytes)
    }
}

impl<T: Number> ParDecoder<T> for T {}
