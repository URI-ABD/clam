//! An adaptation of `Ball` that allows for compression of the dataset.

use std::io::{Read, Write};

use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use num::traits::{FromBytes, ToBytes};
use rayon::prelude::*;

use crate::{cakes::PermutedBall, Cluster, Dataset, DistanceValue, ParCluster, ParDataset, Permutable};

use super::{Decoder, Encoder, ParDecoder, ParEncoder};

/// A `Cluster` for use in compressive search.
#[derive(Clone, bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)]
#[bitcode(recursive)]
#[must_use]
pub struct SquishyBall<T: DistanceValue, S: Cluster<T>> {
    /// The `Cluster` type that the `SquishyBall` is based on.
    source: PermutedBall<T, S>,
    /// The children of the `Cluster`.
    children: Vec<Box<Self>>,
    /// Expected memory cost of recursive compression.
    recursive_cost: usize,
    /// Expected memory cost of flat compression.
    flat_cost: usize,
    /// The minimum expected memory cost of compression.
    minimum_cost: usize,
}

impl<T: DistanceValue, S: Cluster<T>> SquishyBall<T, S> {
    /// Create a new `SquishyBall` from a source `Cluster` tree.
    pub fn from_cluster_tree<I, D, M, Enc, Dec>(root: S, data: &mut D, metric: &M, encoder: &Enc) -> (Self, Vec<usize>)
    where
        D: Permutable<I>,
        M: Fn(&I, &I) -> T,
        Enc: Encoder<I, Dec>,
        Dec: Decoder<I, Enc>,
    {
        let (permuted, permutation) = PermutedBall::from_cluster_tree(root, data);
        let root = Self::adapt_tree_recursive(permuted, data, metric, encoder);
        (root, permutation)
    }

    /// Trims the tree to only include nodes where recursive compression is
    /// cheaper than flat compression.
    pub fn trim(mut self, min_depth: usize) -> Self {
        if self.flat_cost < self.recursive_cost && self.depth() >= min_depth {
            self.children.clear();
        } else if !self.is_leaf() {
            self.children = self
                .children
                .drain(..)
                .map(|child| child.trim(min_depth))
                .map(Box::new)
                .collect();
        }
        self
    }

    /// Returns the offset of the `Cluster`.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.source.offset()
    }

    /// Recursive helper for [`from_cluster_tree`](Self::from_cluster_tree).
    fn adapt_tree_recursive<I, D, M, Enc, Dec>(
        mut source: PermutedBall<T, S>,
        data: &D,
        metric: &M,
        encoder: &Enc,
    ) -> Self
    where
        D: Dataset<I>,
        M: Fn(&I, &I) -> T,
        Enc: Encoder<I, Dec>,
        Dec: Decoder<I, Enc>,
    {
        let center = data.get(source.arg_center());
        let flat_cost = source
            .indices()
            .iter()
            .map(|&i| data.get(i))
            .map(|item| encoder.estimate_delta_size(item, center, metric))
            .sum::<usize>();

        let (children, costs): (Vec<_>, Vec<_>) = source
            .take_children()
            .into_iter()
            .map(|child| {
                let child_center = data.get(child.arg_center());
                let delta_size = encoder.estimate_delta_size(child_center, center, metric);
                let child = Self::adapt_tree_recursive(*child, data, metric, encoder);
                let rec_cost = child.minimum_cost + delta_size;
                (Box::new(child), rec_cost)
            })
            .unzip();

        let recursive_cost = costs.iter().sum();
        let minimum_cost = flat_cost.min(recursive_cost);

        Self {
            source,
            children,
            recursive_cost,
            flat_cost,
            minimum_cost,
        }
    }
}

impl<T: DistanceValue + Send + Sync, S: ParCluster<T>> SquishyBall<T, S> {
    /// Create a new `SquishyBall` from a source `Cluster` tree.
    pub fn par_from_cluster_tree<I, D, M, Enc, Dec>(
        root: S,
        data: &mut D,
        metric: &M,
        encoder: &Enc,
    ) -> (Self, Vec<usize>)
    where
        I: Send + Sync,
        D: ParDataset<I> + Permutable<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        Enc: ParEncoder<I, Dec>,
        Dec: ParDecoder<I, Enc>,
        Enc::Bytes: Send + Sync,
    {
        let (permuted, permutation) = PermutedBall::par_from_cluster_tree(root, data);
        let root = Self::par_adapt_tree_recursive(permuted, data, metric, encoder);
        (root, permutation)
    }

    /// Trims the tree to only include nodes where recursive compression is
    /// cheaper than flat compression.
    pub fn par_trim(mut self, min_depth: usize) -> Self {
        if self.flat_cost < self.recursive_cost && self.depth() >= min_depth {
            self.children.clear();
        } else if !self.is_leaf() {
            self.children = self
                .children
                .drain(..)
                .map(|child| child.trim(min_depth))
                .map(Box::new)
                .collect();
        }
        self
    }

    /// Recursive helper for [`par_from_cluster_tree`](Self::par_from_cluster_tree).
    fn par_adapt_tree_recursive<I, D, M, Enc, Dec>(
        mut source: PermutedBall<T, S>,
        data: &D,
        metric: &M,
        encoder: &Enc,
    ) -> Self
    where
        I: Send + Sync,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        Enc: ParEncoder<I, Dec>,
        Dec: ParDecoder<I, Enc>,
        Enc::Bytes: Send + Sync,
    {
        let center = data.get(source.arg_center());
        let flat_cost = source
            .indices()
            .par_iter()
            .map(|&i| data.get(i))
            .map(|item| encoder.par_estimate_delta_size(item, center, metric))
            .sum::<usize>();

        let (children, costs): (Vec<_>, Vec<_>) = source
            .take_children()
            .into_par_iter()
            .map(|child| {
                let child_center = data.get(child.arg_center());
                let delta_size = encoder.par_estimate_delta_size(child_center, center, metric);
                let child = Self::par_adapt_tree_recursive(*child, data, metric, encoder);
                let rec_cost = child.minimum_cost + delta_size;
                (Box::new(child), rec_cost)
            })
            .unzip();

        let recursive_cost = costs.iter().sum();
        let minimum_cost = flat_cost.min(recursive_cost);

        Self {
            source,
            children,
            recursive_cost,
            flat_cost,
            minimum_cost,
        }
    }
}

impl<T: DistanceValue + core::fmt::Debug, S: Cluster<T> + core::fmt::Debug> core::fmt::Debug for SquishyBall<T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SquishyBall")
            .field("source", &self.source)
            .field("recursive_cost", &self.recursive_cost)
            .field("flat_cost", &self.flat_cost)
            .field("minimum_cost", &self.minimum_cost)
            .field("children", &!self.children.is_empty())
            .finish()
    }
}

impl<T: DistanceValue, S: Cluster<T>> PartialEq for SquishyBall<T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl<T: DistanceValue, S: Cluster<T>> Eq for SquishyBall<T, S> {}

impl<T: DistanceValue, S: Cluster<T>> PartialOrd for SquishyBall<T, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: DistanceValue, S: Cluster<T>> Ord for SquishyBall<T, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.source.cmp(&other.source)
    }
}

impl<T: DistanceValue, S: Cluster<T>> Cluster<T> for SquishyBall<T, S> {
    fn depth(&self) -> usize {
        self.source.depth()
    }

    fn cardinality(&self) -> usize {
        self.source.cardinality()
    }

    fn arg_center(&self) -> usize {
        self.source.arg_center()
    }

    fn set_arg_center(&mut self, arg_center: usize) {
        self.source.set_arg_center(arg_center);
    }

    fn radius(&self) -> T {
        self.source.radius()
    }

    fn arg_radial(&self) -> usize {
        self.source.arg_radial()
    }

    fn set_arg_radial(&mut self, arg_radial: usize) {
        self.source.set_arg_radial(arg_radial);
    }

    fn lfd(&self) -> f32 {
        self.source.lfd()
    }

    fn contains(&self, index: usize) -> bool {
        self.source.contains(index)
    }

    fn indices(&self) -> Vec<usize> {
        self.source.indices()
    }

    fn set_indices(&mut self, indices: &[usize]) {
        self.source.set_indices(indices);
    }

    fn take_indices(&mut self) -> Vec<usize> {
        self.source.take_indices()
    }

    fn children(&self) -> Vec<&Self> {
        self.children.iter().map(AsRef::as_ref).collect()
    }

    fn children_mut(&mut self) -> Vec<&mut Self> {
        self.children.iter_mut().map(AsMut::as_mut).collect()
    }

    fn set_children(&mut self, children: Vec<Box<Self>>) {
        self.children = children;
    }

    fn take_children(&mut self) -> Vec<Box<Self>> {
        core::mem::take(&mut self.children)
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        self.source.is_descendant_of(&other.source)
    }
}

impl<T: DistanceValue + Send + Sync, S: ParCluster<T>> ParCluster<T> for SquishyBall<T, S> {
    fn par_indices(&self) -> impl ParallelIterator<Item = usize> {
        self.source.par_indices()
    }
}

impl<T: DistanceValue + ToBytes<Bytes = Vec<u8>> + FromBytes<Bytes = Vec<u8>>, S: Cluster<T> + crate::DiskIO>
    crate::DiskIO for SquishyBall<T, S>
{
    fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let costs: [[u8; 8]; 3] = [
            self.recursive_cost.to_le_bytes(),
            self.flat_cost.to_le_bytes(),
            self.minimum_cost.to_le_bytes(),
        ];
        let members: (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) = (
            self.source.to_bytes()?,
            bitcode::encode(&costs).map_err(|e| e.to_string())?,
            self.children
                .iter()
                .map(|c| c.to_bytes())
                .collect::<Result<Vec<_>, _>>()?,
        );
        bitcode::encode(&members).map_err(|e| e.to_string())
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (source_bytes, costs_bytes, children_bytes): (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) =
            bitcode::decode(bytes).map_err(|e| e.to_string())?;

        let source = PermutedBall::<T, S>::from_bytes(&source_bytes)?;
        let [recursive_bytes, unitary_bytes, minimum_bytes]: [[u8; 8]; 3] =
            bitcode::decode(&costs_bytes).map_err(|e| e.to_string())?;
        let recursive_cost = usize::from_le_bytes(recursive_bytes);
        let flat_cost = usize::from_le_bytes(unitary_bytes);
        let minimum_cost = usize::from_le_bytes(minimum_bytes);

        let children = children_bytes
            .into_iter()
            .map(|b| Self::from_bytes(&b).map(Box::new))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            source,
            children,
            recursive_cost,
            flat_cost,
            minimum_cost,
        })
    }

    fn write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&self.to_bytes()?).map_err(|e| e.to_string())?;
        let bytes = encoder.finish().map_err(|e| e.to_string())?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    fn read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        let mut bytes = Vec::new();
        let mut decoder = GzDecoder::new(std::fs::File::open(path).map_err(|e| e.to_string())?);
        decoder.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        Self::from_bytes(&bytes)
    }
}

impl<
        T: DistanceValue + ToBytes<Bytes = Vec<u8>> + FromBytes<Bytes = Vec<u8>> + Send + Sync,
        S: ParCluster<T> + crate::ParDiskIO,
    > crate::ParDiskIO for SquishyBall<T, S>
{
    fn par_to_bytes(&self) -> Result<Vec<u8>, String> {
        let costs: [[u8; 8]; 3] = [
            self.recursive_cost.to_le_bytes(),
            self.flat_cost.to_le_bytes(),
            self.minimum_cost.to_le_bytes(),
        ];
        let members: (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) = (
            self.source.par_to_bytes()?,
            bitcode::encode(&costs).map_err(|e| e.to_string())?,
            self.children
                .par_iter()
                .map(|c| c.par_to_bytes())
                .collect::<Result<Vec<_>, _>>()?,
        );
        bitcode::encode(&members).map_err(|e| e.to_string())
    }

    fn par_from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (source_bytes, costs_bytes, children_bytes): (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) =
            bitcode::decode(bytes).map_err(|e| e.to_string())?;

        let source = PermutedBall::<T, S>::par_from_bytes(&source_bytes)?;
        let [recursive_bytes, unitary_bytes, minimum_bytes]: [[u8; 8]; 3] =
            bitcode::decode(&costs_bytes).map_err(|e| e.to_string())?;
        let recursive_cost = usize::from_le_bytes(recursive_bytes);
        let flat_cost = usize::from_le_bytes(unitary_bytes);
        let minimum_cost = usize::from_le_bytes(minimum_bytes);

        let children = children_bytes
            .into_par_iter()
            .map(|b| Self::par_from_bytes(&b).map(Box::new))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            source,
            children,
            recursive_cost,
            flat_cost,
            minimum_cost,
        })
    }

    fn par_write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&self.par_to_bytes()?).map_err(|e| e.to_string())?;
        let bytes = encoder.finish().map_err(|e| e.to_string())?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    fn par_read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        let mut bytes = Vec::new();
        let mut decoder = GzDecoder::new(std::fs::File::open(path).map_err(|e| e.to_string())?);
        decoder.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        Self::par_from_bytes(&bytes)
    }
}
