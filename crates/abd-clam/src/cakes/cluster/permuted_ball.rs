//! An adaptation of `Ball` that stores indices after reordering the dataset.

use num::traits::{FromBytes, ToBytes};
use rayon::prelude::*;

use crate::{Adapted, Cluster, DistanceValue, ParCluster, ParDataset, Permutable};

/// A `Cluster` that stores indices after reordering the dataset.
///
/// # Type parameters
///
/// - `T`: The type of the distance values.
/// - `S`: The `Cluster` type that the `PermutedBall` is based on.
#[derive(Clone, bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)]
#[bitcode(recursive)]
#[must_use]
pub struct PermutedBall<T: DistanceValue, S: Cluster<T>> {
    /// The `Cluster` type that the `PermutedBall` is based on.
    source: S,
    /// The children of the `Cluster`.
    children: Vec<Box<Self>>,
    /// The parameters of the `Cluster`.
    offset: usize,
    /// Ghosts in the machine.
    phantom: core::marker::PhantomData<T>,
}

impl<T: DistanceValue, S: Cluster<T>> PermutedBall<T, S> {
    /// Creates a new `PermutedBall` tree from a source `Cluster` tree and
    /// reorders the dataset in place.
    pub fn from_cluster_tree<I, D: Permutable<I>>(root: S, data: &mut D) -> (Self, Vec<usize>) {
        let (root, permutation) = Self::adapt_tree_recursive(root, 0);
        data.permute(&permutation);
        (root, permutation)
    }

    /// Returns the offset of the `Cluster`.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Returns an iterator over the indices of the `Cluster`.
    pub fn iter_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.offset..(self.offset + self.cardinality())
    }

    /// Recursive helper for [`from_cluster_tree`](Self::from_cluster_tree).
    fn adapt_tree_recursive(mut source: S, offset: usize) -> (Self, Vec<usize>) {
        let (children, indices) = if source.is_leaf() {
            (vec![], source.take_indices())
        } else {
            let children = source.take_children();
            let child_offsets = children
                .iter()
                .scan(offset, |state, child| {
                    let current = *state;
                    *state += child.cardinality();
                    Some(current)
                })
                .collect::<Vec<_>>();

            let (children, child_indices): (Vec<_>, Vec<_>) = children
                .into_iter()
                .zip(child_offsets)
                .map(|(child, off)| Self::adapt_tree_recursive(*child, off))
                .map(|(c, indices)| (Box::new(c), indices))
                .unzip();

            let indices = child_indices.into_iter().flatten().collect();

            (children, indices)
        };

        source.set_arg_center(new_index(source.arg_center(), &indices, offset));
        source.set_arg_radial(new_index(source.arg_radial(), &indices, offset));

        let c = Self {
            source,
            children,
            offset,
            phantom: core::marker::PhantomData,
        };
        (c, indices)
    }
}

impl<T: DistanceValue + Send + Sync, S: ParCluster<T>> PermutedBall<T, S> {
    /// Creates a new `PermutedBall` tree from a source `Cluster` tree and
    /// reorders the dataset in place.
    pub fn par_from_cluster_tree<I: Send + Sync, D: ParDataset<I> + Permutable<I>>(
        root: S,
        data: &mut D,
    ) -> (Self, Vec<usize>) {
        let (root, permutation) = Self::par_adapt_tree_recursive(root, 0);
        data.permute(&permutation);
        (root, permutation)
    }

    /// Recursive helper for [`par_from_cluster_tree`](Self::par_from_cluster_tree).
    fn par_adapt_tree_recursive(mut source: S, offset: usize) -> (Self, Vec<usize>) {
        let (children, indices) = if source.is_leaf() {
            (vec![], source.take_indices())
        } else {
            let children = source.take_children();
            let child_offsets = children
                .iter()
                .scan(offset, |state, child| {
                    let current = *state;
                    *state += child.cardinality();
                    Some(current)
                })
                .collect::<Vec<_>>();

            let (children, child_indices): (Vec<_>, Vec<_>) = children
                .into_par_iter()
                .zip(child_offsets)
                .map(|(child, off)| Self::par_adapt_tree_recursive(*child, off))
                .map(|(c, indices)| (Box::new(c), indices))
                .unzip();

            let indices = child_indices.into_iter().flatten().collect();

            (children, indices)
        };

        source.set_arg_center(new_index(source.arg_center(), &indices, offset));
        source.set_arg_radial(new_index(source.arg_radial(), &indices, offset));

        let c = Self {
            source,
            children,
            offset,
            phantom: core::marker::PhantomData,
        };
        (c, indices)
    }
}

/// Helper for computing a new index after permutation of data.
fn new_index(i: usize, indices: &[usize], offset: usize) -> usize {
    offset
        + indices
            .iter()
            .position(|x| *x == i)
            .unwrap_or_else(|| unreachable!("i: {i}, indices: {indices:?}, offset: {offset}"))
}

impl<T: DistanceValue, S: Cluster<T> + core::fmt::Debug> core::fmt::Debug for PermutedBall<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PermutedBall")
            .field("source", &self.source)
            .field("offset", &self.offset)
            .field("children", &!self.children.is_empty())
            .finish()
    }
}

impl<T: DistanceValue, S: Cluster<T>> PartialEq for PermutedBall<T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.offset == other.offset && self.cardinality() == other.cardinality()
    }
}

impl<T: DistanceValue, S: Cluster<T>> Eq for PermutedBall<T, S> {}

impl<T: DistanceValue, S: Cluster<T>> PartialOrd for PermutedBall<T, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: DistanceValue, S: Cluster<T>> Ord for PermutedBall<T, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.offset
            .cmp(&self.offset)
            .then_with(|| other.cardinality().cmp(&self.cardinality()))
    }
}

impl<T: DistanceValue, S: Cluster<T>> Cluster<T> for PermutedBall<T, S> {
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
        (self.offset..(self.offset + self.cardinality())).contains(&index)
    }

    fn indices(&self) -> Vec<usize> {
        (self.offset..(self.offset + self.cardinality())).collect()
    }

    fn set_indices(&mut self, indices: &[usize]) {
        let offset = indices[0];
        self.offset = offset;
    }

    fn take_indices(&mut self) -> Vec<usize> {
        self.indices()
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
        std::mem::take(&mut self.children)
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        let range = self.offset..(self.offset + other.cardinality());
        range.contains(&self.offset()) && self.cardinality() <= other.cardinality()
    }
}

impl<T: DistanceValue + Send + Sync, S: ParCluster<T>> ParCluster<T> for PermutedBall<T, S> {
    fn par_indices(&self) -> impl ParallelIterator<Item = usize> {
        (self.offset..(self.offset + self.cardinality())).into_par_iter()
    }
}

impl<T: DistanceValue, S: Cluster<T>> Adapted<T, S> for PermutedBall<T, S> {
    fn source(&self) -> &S {
        &self.source
    }

    fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    fn take_source(self) -> S {
        self.source
    }
}

impl<T: DistanceValue + ToBytes<Bytes = Vec<u8>> + FromBytes<Bytes = Vec<u8>>, S: Cluster<T> + crate::DiskIO>
    crate::DiskIO for PermutedBall<T, S>
{
    fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let members: (Vec<u8>, Vec<Vec<u8>>, usize) = (
            self.source.to_bytes()?,
            self.children
                .iter()
                .map(|child| child.to_bytes())
                .collect::<Result<Vec<_>, _>>()?,
            self.offset,
        );

        bitcode::encode(&members).map_err(|e| e.to_string())
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (source_bytes, children_bytes, offset): (Vec<u8>, Vec<Vec<u8>>, usize) =
            bitcode::decode(bytes).map_err(|e| e.to_string())?;
        let source = S::from_bytes(&source_bytes)?;
        let children = children_bytes
            .into_iter()
            .map(|bytes| Self::from_bytes(&bytes).map(Box::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            source,
            children,
            offset,
            phantom: core::marker::PhantomData,
        })
    }
}

impl<
        T: DistanceValue + ToBytes<Bytes = Vec<u8>> + FromBytes<Bytes = Vec<u8>> + Send + Sync,
        S: ParCluster<T> + crate::ParDiskIO,
    > crate::ParDiskIO for PermutedBall<T, S>
{
    fn par_to_bytes(&self) -> Result<Vec<u8>, String> {
        let members: (Vec<u8>, Vec<Vec<u8>>, usize) = (
            self.source.par_to_bytes()?,
            self.children
                .par_iter()
                .map(|child| child.par_to_bytes())
                .collect::<Result<Vec<_>, _>>()?,
            self.offset,
        );

        bitcode::encode(&members).map_err(|e| e.to_string())
    }

    fn par_from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (source_bytes, children_bytes, offset): (Vec<u8>, Vec<Vec<u8>>, usize) =
            bitcode::decode(bytes).map_err(|e| e.to_string())?;
        let source = S::par_from_bytes(&source_bytes)?;
        let children = children_bytes
            .into_par_iter()
            .map(|bytes| Self::par_from_bytes(&bytes).map(Box::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            source,
            children,
            offset,
            phantom: core::marker::PhantomData,
        })
    }
}
