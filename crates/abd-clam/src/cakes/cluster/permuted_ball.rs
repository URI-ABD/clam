//! An adaptation of `Ball` that stores indices after reordering the dataset.

use distances::Number;
use rayon::prelude::*;

use crate::{
    cluster::{
        adapter::{Adapter, BallAdapter, ParAdapter, ParBallAdapter, ParParams, Params},
        ParCluster,
    },
    dataset::{ParDataset, Permutable},
    metric::ParMetric,
    Ball, Cluster, Dataset, Metric,
};

/// A `Cluster` that stores indices after reordering the dataset.
///
/// # Type parameters
///
/// - `T`: The type of the distance values.
/// - `S`: The `Cluster` type that the `PermutedBall` is based on.
#[derive(Clone)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "disk-io", bitcode(recursive))]
pub struct PermutedBall<T: Number, S: Cluster<T>> {
    /// The `Cluster` type that the `PermutedBall` is based on.
    source: S,
    /// The children of the `Cluster`.
    children: Vec<Box<Self>>,
    /// The parameters of the `Cluster`.
    params: Offset,
    /// Ghosts in the machine.
    phantom: core::marker::PhantomData<T>,
}

impl<T: Number, S: Cluster<T>> PermutedBall<T, S> {
    /// Clears the indices of the source `Cluster` and its children.
    pub fn clear_source_indices(&mut self) {
        self.source.clear_indices();
        if !self.is_leaf() {
            self.children_mut().into_iter().for_each(Self::clear_source_indices);
        }
    }

    /// Returns an iterator over the indices.
    pub fn iter_indices(&self) -> impl Iterator<Item = usize> {
        self.params.offset..(self.params.offset + self.cardinality())
    }
}

impl<T: Number, S: Cluster<T> + core::fmt::Debug> core::fmt::Debug for PermutedBall<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PermutedBall")
            .field("source", &self.source)
            .field("offset", &self.params.offset)
            .field("children", &!self.children.is_empty())
            .finish()
    }
}

impl<T: Number, S: Cluster<T>> PartialEq for PermutedBall<T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.params.offset == other.params.offset && self.cardinality() == other.cardinality()
    }
}

impl<T: Number, S: Cluster<T>> Eq for PermutedBall<T, S> {}

impl<T: Number, S: Cluster<T>> PartialOrd for PermutedBall<T, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Number, S: Cluster<T>> Ord for PermutedBall<T, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.params
            .offset
            .cmp(&other.params.offset)
            .then_with(|| other.cardinality().cmp(&self.cardinality()))
    }
}

impl<T: Number, S: Cluster<T>> std::hash::Hash for PermutedBall<T, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.params.offset, self.cardinality()).hash(state);
    }
}

impl<T: Number, S: Cluster<T>> PermutedBall<T, S> {
    /// Returns the offset of the `Cluster`.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.params.offset
    }
}

impl<T: Number, S: Cluster<T>> Cluster<T> for PermutedBall<T, S> {
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
        (self.params.offset..(self.params.offset + self.cardinality())).contains(&index)
    }

    fn indices(&self) -> Vec<usize> {
        self.iter_indices().collect()
    }

    fn set_indices(&mut self, indices: &[usize]) {
        let offset = indices[0];
        self.params.offset = offset;
    }

    fn extents(&self) -> &[(usize, T)] {
        self.source.extents()
    }

    fn extents_mut(&mut self) -> &mut [(usize, T)] {
        self.source.extents_mut()
    }

    fn add_extent(&mut self, idx: usize, extent: T) {
        self.source.add_extent(idx, extent);
    }

    fn take_extents(&mut self) -> Vec<(usize, T)> {
        self.source.take_extents()
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
        let range = other.params.offset..(other.params.offset + other.cardinality());
        range.contains(&self.offset()) && self.cardinality() <= other.cardinality()
    }
}

impl<T: Number, S: ParCluster<T>> ParCluster<T> for PermutedBall<T, S> {
    fn par_indices(&self) -> impl ParallelIterator<Item = usize> {
        (self.params.offset..(self.params.offset + self.cardinality())).into_par_iter()
    }
}

/// Parameters for adapting the `PermutedBall`.
#[derive(Debug, Default, Copy, Clone)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)
)]
pub struct Offset {
    /// The offset of the slice of indices of the `Cluster` in the reordered
    /// dataset.
    offset: usize,
}

impl<I, T: Number, D: Dataset<I>, S: Cluster<T>> Params<I, T, D, S> for Offset {
    fn child_params<M: Metric<I, T>>(&self, children: &[S], _: &D, _: &M) -> Vec<Self> {
        let mut offset = self.offset;
        children
            .iter()
            .map(|child| {
                let params = Self { offset };
                offset += child.cardinality();
                params
            })
            .collect()
    }
}

impl<I: Send + Sync, T: Number, D: ParDataset<I>, S: ParCluster<T>> ParParams<I, T, D, S> for Offset {
    fn par_child_params<M: ParMetric<I, T>>(&self, children: &[S], data: &D, metric: &M) -> Vec<Self> {
        // Since we need to keep track of the offset, we cannot parallelize this.
        self.child_params(children, data, metric)
    }
}

impl<I, T: Number, D: Dataset<I> + Permutable> BallAdapter<I, T, D, D, Offset> for PermutedBall<T, Ball<T>> {
    /// Creates a new `PermutedBall` tree from a `Ball` tree.
    fn from_ball_tree<M: Metric<I, T>>(ball: Ball<T>, mut data: D, metric: &M) -> (Self, D) {
        let mut root = Self::adapt_tree_iterative(ball, None, &data, metric);
        data.permute(&root.source.indices());
        root.clear_source_indices();
        (root, data)
    }
}

impl<I: Send + Sync, T: Number, D: ParDataset<I> + Permutable> ParBallAdapter<I, T, D, D, Offset>
    for PermutedBall<T, Ball<T>>
{
    /// Creates a new `PermutedBall` tree from a `Ball` tree.
    fn par_from_ball_tree<M: ParMetric<I, T>>(ball: Ball<T>, mut data: D, metric: &M) -> (Self, D) {
        let mut root = Self::par_adapt_tree_iterative(ball, None, &data, metric);
        data.permute(&root.source.indices());
        root.clear_source_indices();
        (root, data)
    }
}

impl<I, T: Number, D: Dataset<I> + Permutable, S: Cluster<T>> Adapter<I, T, D, D, S, Offset> for PermutedBall<T, S> {
    fn new_adapted<M: Metric<I, T>>(source: S, children: Vec<Box<Self>>, params: Offset, _: &D, _: &M) -> Self {
        Self {
            source,
            params,
            children,
            phantom: core::marker::PhantomData,
        }
    }

    fn post_traversal(&mut self) {
        // Update the indices of the important items in the `Cluster`.
        let offset = self.params.offset;
        let indices = self.source.indices();
        self.set_arg_center(new_index(self.source.arg_center(), &indices, offset));
        self.set_arg_radial(new_index(self.source.arg_radial(), &indices, offset));
        for (i, _) in self.extents_mut() {
            *i = new_index(*i, &indices, offset);
        }
    }

    fn source(&self) -> &S {
        &self.source
    }

    fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    fn take_source(self) -> S {
        self.source
    }

    fn params(&self) -> &Offset {
        &self.params
    }
}

/// Helper for computing a new index after permutation of data.
fn new_index(i: usize, indices: &[usize], offset: usize) -> usize {
    offset
        + indices
            .iter()
            .position(|x| *x == i)
            .unwrap_or_else(|| unreachable!("This is a private function and we always pass a valid item."))
}

impl<I: Send + Sync, T: Number, D: ParDataset<I> + Permutable, S: ParCluster<T>> ParAdapter<I, T, D, D, S, Offset>
    for PermutedBall<T, S>
{
    fn par_new_adapted<M: ParMetric<I, T>>(
        source: S,
        children: Vec<Box<Self>>,
        params: Offset,
        data: &D,
        metric: &M,
    ) -> Self {
        Self::new_adapted(source, children, params, data, metric)
    }
}

#[cfg(feature = "disk-io")]
impl<T: Number, S: crate::cluster::Csv<T>> crate::cluster::Csv<T> for PermutedBall<T, S> {
    fn header(&self) -> Vec<String> {
        let mut header = self.source.header();
        header.push("offset".to_string());
        header
    }

    fn row(&self) -> Vec<String> {
        let mut row = self.source.row();
        row.pop();
        row.extend(vec![
            self.children.is_empty().to_string(),
            self.params.offset.to_string(),
        ]);
        row
    }
}

#[cfg(feature = "disk-io")]
impl<T: Number, S: crate::cluster::ParCsv<T>> crate::cluster::ParCsv<T> for PermutedBall<T, S> {}

#[cfg(feature = "disk-io")]
impl<T: Number, S: crate::cluster::ClusterIO<T>> crate::cluster::ClusterIO<T> for PermutedBall<T, S> {}

#[cfg(feature = "disk-io")]
impl<T: Number, S: crate::cluster::ParClusterIO<T>> crate::cluster::ParClusterIO<T> for PermutedBall<T, S> {}
