//! An adaptation of `Ball` that allows for compression of the dataset and
//! search in the compressed space.

use distances::Number;
use rayon::prelude::*;

use crate::{
    adapters::{Adapted, Adapter, BallAdapter, ParAdapter, ParBallAdapter, ParParams, Params},
    cluster::ParCluster,
    dataset::{ParDataset, Permutable},
    metric::ParMetric,
    Ball, Cluster, Dataset, Metric,
};

#[cfg(feature = "disk-io")]
use std::io::{Read, Write};

#[cfg(feature = "disk-io")]
use flate2::{read::GzDecoder, write::GzEncoder, Compression};

/// A `Cluster` for use in compressive search.
#[derive(Clone)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "disk-io", bitcode(recursive))]
#[must_use]
pub struct SquishyBall<T: Number, S: Cluster<T>> {
    /// The `Cluster` type that the `SquishyBall` is based on.
    source: S,
    /// Parameters for the `SquishyBall`.
    costs: SquishCosts<T>,
    /// The children of the `Cluster`.
    children: Vec<Box<Self>>,
}

impl<T: Number, S: Cluster<T> + core::fmt::Debug> core::fmt::Debug for SquishyBall<T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SquishyBall")
            .field("source", &self.source)
            .field("recursive_cost", &self.costs.recursive)
            .field("unitary_cost", &self.costs.unitary)
            .field("minimum_cost", &self.costs.minimum)
            .field("children", &!self.children.is_empty())
            .finish()
    }
}

impl<T: Number, S: Cluster<T>> SquishyBall<T, S> {
    /// Get the unitary cost of the `SquishyBall`.
    pub const fn unitary_cost(&self) -> T {
        self.costs.unitary
    }

    /// Get the recursive cost of the `SquishyBall`.
    pub const fn recursive_cost(&self) -> T {
        self.costs.recursive
    }

    /// Trims the tree by removing empty children of clusters whose unitary cost
    /// is greater than the recursive cost.
    pub fn trim(&mut self, min_depth: usize) {
        if !self.children.is_empty() {
            if (self.costs.unitary <= self.costs.recursive) && (self.depth() >= min_depth) {
                self.children.clear();
            } else {
                self.children.iter_mut().for_each(|c| c.trim(min_depth));
            }
        }
    }

    /// Sets the costs for the tree.
    pub fn set_costs<I, D: Dataset<I>, M: Metric<I, T>>(&mut self, data: &D, metric: &M) {
        self.set_unitary_cost(data, metric);
        if self.children.is_empty() {
            self.costs.recursive = T::ZERO;
        } else {
            self.children.iter_mut().for_each(|c| c.set_costs(data, metric));
            self.set_recursive_cost(data, metric);
        }
        self.set_min_cost();
    }

    /// Calculates the unitary cost of the `Cluster`.
    fn set_unitary_cost<I, D: Dataset<I>, M: Metric<I, T>>(&mut self, data: &D, metric: &M) {
        self.costs.unitary = self
            .source
            .indices()
            .iter()
            .map(|&i| data.one_to_one(i, self.arg_center(), metric))
            .sum();
    }

    /// Calculates the recursive cost of the `Cluster`.
    fn set_recursive_cost<I, D: Dataset<I>, M: Metric<I, T>>(&mut self, data: &D, metric: &M) {
        if self.children.is_empty() {
            self.costs.recursive = T::ZERO;
        } else {
            let children = self.children();
            let child_costs = children.iter().map(|c| c.costs.minimum).sum::<T>();
            let child_centers = children.iter().map(|c| c.arg_center()).collect::<Vec<_>>();
            self.costs.recursive = child_costs
                + data
                    .one_to_many(self.arg_center(), &child_centers, metric)
                    .map(|(_, d)| d)
                    .sum::<T>();
        }
    }

    /// Sets the minimum cost of the `Cluster`.
    fn set_min_cost(&mut self) {
        self.costs.minimum = if self.costs.recursive < self.costs.unitary {
            self.costs.recursive
        } else {
            self.costs.unitary
        };
    }
}

impl<T: Number, S: ParCluster<T>> SquishyBall<T, S> {
    /// Sets the costs for the tree.
    pub fn par_set_costs<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>>(&mut self, data: &D, metric: &M) {
        self.par_set_unitary_cost(data, metric);
        if self.children.is_empty() {
            self.costs.recursive = T::ZERO;
        } else {
            self.children.par_iter_mut().for_each(|c| c.par_set_costs(data, metric));
            self.par_set_recursive_cost(data, metric);
        }
        self.set_min_cost();
    }

    /// Calculates the unitary cost of the `Cluster`.
    fn par_set_unitary_cost<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>>(&mut self, data: &D, metric: &M) {
        self.costs.unitary = self
            .par_indices()
            .map(|i| data.par_one_to_one(i, self.arg_center(), metric))
            .sum();
    }

    /// Calculates the recursive cost of the `Cluster`.
    fn par_set_recursive_cost<I: Send + Sync, D: ParDataset<I>, M: ParMetric<I, T>>(&mut self, data: &D, metric: &M) {
        if self.children.is_empty() {
            self.costs.recursive = T::ZERO;
        } else {
            let children = self.children();
            let child_costs = children.iter().map(|c| c.costs.minimum).sum::<T>();
            let child_centers = children.iter().map(|c| c.arg_center()).collect::<Vec<_>>();
            self.costs.recursive = child_costs
                + data
                    .par_one_to_many(self.arg_center(), &child_centers, metric)
                    .map(|(_, d)| d)
                    .sum();
        }
    }
}

impl<T: Number, S: Cluster<T>> Cluster<T> for SquishyBall<T, S> {
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

    fn extents(&self) -> &[(usize, T)] {
        &self.source.extents()[..1]
    }

    fn extents_mut(&mut self) -> &mut [(usize, T)] {
        &mut self.source.extents_mut()[..1]
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
        core::mem::take(&mut self.children)
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        self.source.is_descendant_of(&other.source)
    }
}

impl<T: Number, S: ParCluster<T>> ParCluster<T> for SquishyBall<T, S> {
    fn par_indices(&self) -> impl ParallelIterator<Item = usize> {
        self.source.par_indices()
    }
}

impl<T: Number, S: Cluster<T>> Adapted<T, S> for SquishyBall<T, S> {
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

/// Parameters for the `OffsetBall`.
#[derive(Debug, Default, Copy, Clone)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)
)]
pub struct SquishCosts<T> {
    /// Expected memory cost of recursive compression.
    recursive: T,
    /// Expected memory cost of unitary compression.
    unitary: T,
    /// The minimum expected memory cost of compression.
    minimum: T,
}

impl<I, T: Number, D: Dataset<I>, S: Cluster<T>, M: Metric<I, T>> Params<I, T, D, S, M> for SquishCosts<T> {
    fn child_params(&self, children: &[S], _: &D, _: &M) -> Vec<Self> {
        children.iter().map(|_| Self::default()).collect()
    }
}

impl<I: Send + Sync, T: Number, D: ParDataset<I>, S: ParCluster<T>, M: ParMetric<I, T>> ParParams<I, T, D, S, M>
    for SquishCosts<T>
{
    fn par_child_params(&self, children: &[S], data: &D, metric: &M) -> Vec<Self> {
        self.child_params(children, data, metric)
    }
}

impl<I: Clone, T: Number, D: Dataset<I> + Permutable, M: Metric<I, T>> BallAdapter<I, T, D, D, M, SquishCosts<T>>
    for SquishyBall<T, Ball<T>>
{
    fn from_ball_tree(ball: Ball<T>, data: D, metric: &M) -> (Self, D) {
        let mut root =
            <Self as Adapter<I, T, D, D, Ball<T>, M, SquishCosts<T>>>::adapt_tree_iterative(ball, None, &data, metric);
        root.set_costs(&data, metric);
        root.trim(4);
        (root, data)
    }
}

impl<I: Clone + Send + Sync, T: Number, D: ParDataset<I> + Permutable, M: ParMetric<I, T>>
    ParBallAdapter<I, T, D, D, M, SquishCosts<T>> for SquishyBall<T, Ball<T>>
{
    fn par_from_ball_tree(ball: Ball<T>, data: D, metric: &M) -> (Self, D) {
        let mut root = <Self as ParAdapter<I, T, D, D, Ball<T>, M, SquishCosts<T>>>::par_adapt_tree_iterative(
            ball, None, &data, metric,
        );
        root.par_set_costs(&data, metric);
        root.trim(4);
        (root, data)
    }
}

impl<I, T: Number, Co: Dataset<I>, Dec: Dataset<I>, S: Cluster<T>, M: Metric<I, T>>
    Adapter<I, T, Co, Dec, S, M, SquishCosts<T>> for SquishyBall<T, S>
{
    fn new_adapted(source: S, children: Vec<Box<Self>>, params: SquishCosts<T>, _: &Co, _: &M) -> Self {
        Self {
            source,
            costs: params,
            children,
        }
    }

    fn post_traversal(&mut self) {}

    fn params(&self) -> &SquishCosts<T> {
        &self.costs
    }
}

impl<I: Send + Sync, T: Number, Co: ParDataset<I>, Dec: ParDataset<I>, S: ParCluster<T>, M: ParMetric<I, T>>
    ParAdapter<I, T, Co, Dec, S, M, SquishCosts<T>> for SquishyBall<T, S>
{
    fn par_new_adapted(source: S, children: Vec<Box<Self>>, params: SquishCosts<T>, _: &Co, _: &M) -> Self {
        Self {
            source,
            costs: params,
            children,
        }
    }
}

impl<T: Number, S: Cluster<T>> PartialEq for SquishyBall<T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl<T: Number, S: Cluster<T>> Eq for SquishyBall<T, S> {}

impl<T: Number, S: Cluster<T>> PartialOrd for SquishyBall<T, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Number, S: Cluster<T>> Ord for SquishyBall<T, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.source.cmp(&other.source)
    }
}

impl<T: Number, S: Cluster<T>> core::hash::Hash for SquishyBall<T, S> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
    }
}

#[cfg(feature = "disk-io")]
impl<T: Number, S: crate::cluster::Csv<T>> crate::cluster::Csv<T> for SquishyBall<T, S> {
    fn header(&self) -> Vec<String> {
        let mut header = self.source.header();
        header.extend(vec![
            "recursive_cost".to_string(),
            "unitary_cost".to_string(),
            "minimum_cost".to_string(),
        ]);
        header
    }

    fn row(&self) -> Vec<String> {
        let mut row = self.source.row();
        row.pop();
        row.extend(vec![
            self.children.is_empty().to_string(),
            self.costs.recursive.to_string(),
            self.costs.unitary.to_string(),
            self.costs.minimum.to_string(),
        ]);
        row
    }
}

#[cfg(feature = "disk-io")]
impl<T: Number, S: crate::cluster::ParCsv<T>> crate::cluster::ParCsv<T> for SquishyBall<T, S> {}

#[cfg(feature = "disk-io")]
impl<T: Number + bitcode::Encode + bitcode::Decode, S: Cluster<T> + crate::DiskIO> crate::DiskIO for SquishyBall<T, S> {
    fn write_to<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        let bytes = bitcode::encode(self).map_err(|e| e.to_string())?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&bytes).map_err(|e| e.to_string())?;
        let bytes = encoder.finish().map_err(|e| e.to_string())?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    fn read_from<P: AsRef<std::path::Path>>(path: &P) -> Result<Self, String> {
        let mut bytes = Vec::new();
        let mut decoder = GzDecoder::new(std::fs::File::open(path).map_err(|e| e.to_string())?);
        decoder.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        bitcode::decode(&bytes).map_err(|e| e.to_string())
    }
}

#[cfg(feature = "disk-io")]
impl<T: Number + bitcode::Encode + bitcode::Decode, S: ParCluster<T> + crate::ParDiskIO> crate::ParDiskIO
    for SquishyBall<T, S>
{
}
