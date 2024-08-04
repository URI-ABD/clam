//! Traits that define the behavior of compression algorithms.

use distances::Number;

use crate::{Cluster, Dataset};

/// A trait that defines how a value can be encoded in terms of a reference.
pub trait Encodable: Clone {
    /// Converts the value to a byte array.
    fn as_bytes(&self) -> Box<[u8]>;

    /// Encodes the value in terms of a reference.
    fn encode(&self, reference: &Self) -> Box<[u8]>;
}

/// A trait that defines how a dataset can be compressed.
pub trait Compressible<I: Encodable, U: Number>: Dataset<I, U> {
    /// Recursively encodes the centers of the clusters in terms of their
    /// parents' center.
    fn encode_centers<C: Cluster<U>>(&self, root: &C) -> Box<[u8]> {
        let mut bytes = Vec::new();

        let root_center = self.get(root.arg_center()).as_bytes();
        bytes.extend_from_slice(&root_center.len().to_le_bytes());
        bytes.extend_from_slice(&root_center);

        for (reference_index, target_index) in index_pairs(root) {
            bytes.extend_from_slice(&target_index.to_le_bytes());
            bytes.extend_from_slice(&reference_index.to_le_bytes());

            let encoding = self.get(target_index).encode(self.get(reference_index));

            bytes.extend_from_slice(&encoding.len().to_le_bytes());
            bytes.extend_from_slice(&encoding);
        }

        bytes.into_boxed_slice()
    }

    /// Encodes all the instances of leaf clusters in terms of their centers.
    ///
    /// # Returns
    ///
    /// - A flattened vector of encoded instances.
    /// - A vector of offsets that indicate the start of the instances for each
    ///   leaf cluster in the flattened vector.
    fn encode_leaves<C: Cluster<U>>(&self, root: &C) -> (Box<[u8]>, Vec<usize>) {
        let mut offsets = Vec::new();
        let mut bytes = Vec::new();

        for leaf in root.leaves() {
            offsets.push(bytes.len());

            bytes.extend_from_slice(&leaf.cardinality().to_le_bytes());
            let center = self.get(leaf.arg_center());
            for i in leaf.indices() {
                let encoding = self.get(i).encode(center);
                bytes.extend_from_slice(&encoding.len().to_le_bytes());
                bytes.extend_from_slice(&encoding);
            }
        }

        (bytes.into_boxed_slice(), offsets)
    }
}

/// Recursively finds the pairs of indices that represent the parent and child
/// centers of a cluster.
fn index_pairs<U: Number, C: Cluster<U>>(c: &C) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let center = c.arg_center();
    for child in c.child_clusters() {
        let child_center = child.arg_center();
        pairs.push((center, child_center));
    }
    for child in c.child_clusters() {
        pairs.extend(index_pairs(child));
    }
    pairs
}
