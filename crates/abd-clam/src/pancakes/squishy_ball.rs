//! An adaptation of `Ball` that allows for compression of the dataset.

use rayon::prelude::*;

use crate::{cakes::PermutedBall, Cluster, Dataset, DatasetMut, DistanceValue, ParCluster};

use super::{Decoder, Encoder};

/// A `Cluster` for use in compressive search.
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
    pub fn from_cluster_tree<I, D, Enc, Dec>(root: S, data: &mut D, encoder: &Enc) -> (Self, Vec<usize>)
    where
        D: DatasetMut<I>,
        Enc: Encoder<I, Dec>,
        Dec: Decoder<I, Enc>,
    {
        let (permuted, permutation) = PermutedBall::from_cluster_tree(root, data);
        let root = Self::adapt_tree_recursive(permuted, data, encoder);
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
    fn adapt_tree_recursive<I, D, Enc, Dec>(mut source: PermutedBall<T, S>, data: &D, encoder: &Enc) -> Self
    where
        D: Dataset<I>,
        Enc: Encoder<I, Dec>,
        Dec: Decoder<I, Enc>,
    {
        let center = data.get(source.arg_center());
        let flat_cost = source
            .indices()
            .iter()
            .map(|&i| data.get(i))
            .map(|item| encoder.estimate_delta_size(item, center))
            .sum::<usize>();

        let (children, costs): (Vec<_>, Vec<_>) = source
            .take_children()
            .into_iter()
            .map(|child| {
                let child_center = data.get(child.arg_center());
                let delta_size = encoder.estimate_delta_size(child_center, center);
                let child = Self::adapt_tree_recursive(*child, data, encoder);
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
