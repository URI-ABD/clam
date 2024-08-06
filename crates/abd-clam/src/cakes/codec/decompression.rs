//! Traits and an implementation for decompressing datasets.

use std::collections::HashMap;

use distances::Number;

use crate::{Cluster, Dataset};

/// A trait that defines how a value can be decoded in terms of a reference.
pub trait Decodable {
    /// Decodes the value from a byte array.
    fn from_bytes(bytes: &[u8]) -> Self;

    /// Decodes the value in terms of a reference.
    fn decode(reference: &Self, bytes: &[u8]) -> Self;
}

/// A trait that defines how a dataset can be decompressed.
pub trait Decompressible<I: Decodable, U: Number>: Dataset<I, U> + Sized {
    /// Returns the centers of the clusters in the tree associated with this
    /// dataset.
    fn centers(&self) -> &HashMap<usize, I>;

    /// Returns the bytes slice representing all compressed leaves.
    fn leaf_bytes(&self) -> &[u8];

    /// Returns the offsets of the leaves' encodings in the bytes.
    fn leaf_offsets(&self) -> &[usize];

    /// Decodes the centers of the clusters in terms of their parents' center.
    fn decode_centers<C: Cluster<I, U, Self>>(&self, root: &C, bytes: &[u8]) -> HashMap<usize, I> {
        let mut offset = 0;

        let root_center = I::from_bytes(&super::read_encoding(bytes, &mut offset));

        let mut centers = HashMap::new();
        centers.insert(root.arg_center(), root_center);

        while offset < bytes.len() {
            let target_index = super::read_usize(bytes, &mut offset);
            let reference_index = super::read_usize(bytes, &mut offset);

            let encoding = super::read_encoding(bytes, &mut offset);
            let reference = &centers[&reference_index];
            let target = I::decode(reference, &encoding);

            centers.insert(target_index, target);
        }

        centers
    }

    /// Decodes all the instances of a leaf cluster in terms of its center.
    fn decode_leaf(&self, mut offset: usize) -> Vec<I> {
        let mut instances = Vec::new();

        let arg_center = super::read_usize(self.leaf_bytes(), &mut offset);
        let center = &self.centers()[&arg_center];

        let cardinality = super::read_usize(self.leaf_bytes(), &mut offset);

        for _ in 0..cardinality {
            let encoding = super::read_encoding(self.leaf_bytes(), &mut offset);
            let instance = I::decode(center, &encoding);
            instances.push(instance);
        }

        instances
    }
}
