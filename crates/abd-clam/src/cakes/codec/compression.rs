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
    /// Recursively encodes the centers of the clusters in terms of their
    /// parents' center.
    fn encode_centers<D: Dataset<I, U>, C: Cluster<I, U, D>>(&self, root: &C) -> Box<[u8]> {
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
    /// - A vector of cumulative cardinalities of leaves.
    fn encode_leaves<D: Dataset<I, U>, C: Cluster<I, U, D>>(&self, root: &C) -> (Box<[u8]>, Vec<usize>, Vec<usize>) {
        let mut cumulative_cardinalities = vec![0];
        let mut offsets = Vec::new();
        let mut bytes = Vec::new();

        for (i, leaf) in root.leaves().into_iter().enumerate() {
            cumulative_cardinalities.push(cumulative_cardinalities[i] + leaf.cardinality());
            offsets.push(bytes.len());

            bytes.extend_from_slice(&leaf.arg_center().to_le_bytes());
            bytes.extend_from_slice(&leaf.cardinality().to_le_bytes());
            let center = self.get(leaf.arg_center());
            for i in leaf.indices() {
                let encoding = self.get(i).encode(center);
                bytes.extend_from_slice(&encoding.len().to_le_bytes());
                bytes.extend_from_slice(&encoding);
            }
        }

        (bytes.into_boxed_slice(), offsets, cumulative_cardinalities)
    }
}

/// A trait that defines how a dataset can be compressed.
pub trait ParCompressible<I: Encodable + Send + Sync, U: Number>: Compressible<I, U> + ParDataset<I, U> + Sized {
    /// Recursively encodes the centers of the clusters in terms of their
    /// parents' center.
    fn par_encode_centers<D: ParDataset<I, U>, C: ParCluster<I, U, D>>(&self, root: &C) -> Box<[u8]> {
        let encodings = index_pairs(root)
            .into_par_iter()
            .map(|(r, t)| (r, t, self.get(t).encode(self.get(r))))
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();

        let root_center = self.get(root.arg_center()).as_bytes();
        bytes.extend_from_slice(&root_center.len().to_le_bytes());
        bytes.extend_from_slice(&root_center);

        for (reference, target, enc) in encodings {
            bytes.extend_from_slice(&target.to_le_bytes());
            bytes.extend_from_slice(&reference.to_le_bytes());

            bytes.extend_from_slice(&enc.len().to_le_bytes());
            bytes.extend_from_slice(&enc);
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
    /// - A vector of cumulative cardinalities of leaves.
    fn par_encode_leaves<D: ParDataset<I, U>, C: ParCluster<I, U, D>>(
        &self,
        root: &C,
    ) -> (Box<[u8]>, Vec<usize>, Vec<usize>) {
        let encodings = root
            .leaves()
            .into_par_iter()
            .enumerate()
            .map(|(i, leaf)| {
                let center = self.get(leaf.arg_center());
                let indices = leaf.indices().collect::<Vec<_>>();
                let encodings = indices
                    .into_par_iter()
                    .map(|j| self.get(j).encode(center))
                    .collect::<Vec<_>>();
                (i, leaf.cardinality(), leaf.arg_center(), encodings)
            })
            .collect::<Vec<_>>();

        let mut cumulative_cardinalities = vec![0];
        let mut offsets = Vec::new();
        let mut bytes = Vec::new();

        for (i, car, cen, enc) in encodings {
            cumulative_cardinalities.push(cumulative_cardinalities[i] + car);
            offsets.push(bytes.len());

            bytes.extend_from_slice(&cen.to_le_bytes());
            bytes.extend_from_slice(&car.to_le_bytes());
            for e in enc {
                bytes.extend_from_slice(&e.len().to_le_bytes());
                bytes.extend_from_slice(&e);
            }
        }

        (bytes.into_boxed_slice(), offsets, cumulative_cardinalities)
    }
}

/// Recursively finds the pairs of indices that represent the parent and child
/// centers of a cluster.
fn index_pairs<I, U: Number, D: Dataset<I, U>, C: Cluster<I, U, D>>(c: &C) -> Vec<(usize, usize)> {
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
