//! An adaptation of `Ball` that allows for compression of the dataset and
//! search in the compressed space.

use core::fmt::Debug;

use std::marker::PhantomData;

use distances::Number;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    adapter::{Adapter, BallAdapter, ParAdapter, ParBallAdapter, ParParams, Params},
    cakes::OffBall,
    cluster::ParCluster,
    dataset::{metric_space::ParMetricSpace, ParDataset},
    Ball, Cluster, Dataset, MetricSpace, Permutable,
};

use super::{
    compression::ParCompressible, decompression::ParDecompressible, CodecData, Compressible, Decodable, Decompressible,
    Encodable,
};

/// A variant of `Ball` that stores indices after reordering the dataset.
#[derive(Clone, Serialize, Deserialize)]
pub struct SquishyBall<
    I: Encodable + Decodable,
    U: Number,
    Co: Compressible<I, U>,
    Dec: Decompressible<I, U>,
    S: Cluster<I, U, Co>,
> {
    /// The `Cluster` type that the `OffsetBall` is based on.
    source: OffBall<I, U, Co, S>,
    /// The children of the `Cluster`.
    children: Vec<(usize, U, Box<Self>)>,
    /// Parameters for the `OffsetBall`.
    costs: SquishCosts<U>,
    /// Phantom data to satisfy the compiler.
    _dc: PhantomData<Dec>,
}

impl<
        I: Encodable + Decodable,
        U: Number,
        Co: Compressible<I, U>,
        Dec: Decompressible<I, U>,
        S: Cluster<I, U, Co> + Debug,
    > Debug for SquishyBall<I, U, Co, Dec, S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SquishyBall")
            .field("source", &self.source)
            .field("children", &!self.children.is_empty())
            .field("recursive_cost", &self.costs.recursive)
            .field("unitary_cost", &self.costs.unitary)
            .field("minimum_cost", &self.costs.minimum)
            .finish()
    }
}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, S: Cluster<I, U, Co>, M>
    SquishyBall<I, U, Co, CodecData<I, U, M>, S>
{
    /// Allows for the `SquishyBall` to be use with the same compressed dataset under different metadata types.
    pub fn with_metadata_type<Me>(self) -> SquishyBall<I, U, Co, CodecData<I, U, Me>, S> {
        SquishyBall {
            source: self.source,
            children: self
                .children
                .into_iter()
                .map(|(i, r, c)| (i, r, Box::new(c.with_metadata_type())))
                .collect(),
            costs: self.costs,
            _dc: PhantomData,
        }
    }
}

impl<I: Encodable + Decodable, U: Number, D: Compressible<I, U> + Permutable>
    BallAdapter<I, U, D, CodecData<I, U, usize>, SquishCosts<U>>
    for SquishyBall<I, U, D, CodecData<I, U, usize>, Ball<I, U, D>>
{
    fn from_ball_tree(ball: Ball<I, U, D>, data: D) -> (Self, CodecData<I, U, usize>) {
        let (off_ball, data) = OffBall::from_ball_tree(ball, data);
        let mut root = Self::adapt_tree_iterative(off_ball, None);
        root.set_costs(&data);
        root.trim(4);
        let data = CodecData::from_compressible(&data, &root);
        (root, data)
    }
}

impl<I: Encodable + Decodable + Send + Sync, U: Number, D: ParCompressible<I, U> + Permutable>
    ParBallAdapter<I, U, D, CodecData<I, U, usize>, SquishCosts<U>>
    for SquishyBall<I, U, D, CodecData<I, U, usize>, Ball<I, U, D>>
{
    fn par_from_ball_tree(ball: Ball<I, U, D>, data: D) -> (Self, CodecData<I, U, usize>) {
        let (off_ball, data) = OffBall::par_from_ball_tree(ball, data);
        let mut root = Self::par_adapt_tree_iterative(off_ball, None);
        root.par_set_costs(&data);
        root.trim(4);
        let data = CodecData::par_from_compressible(&data, &root);
        (root, data)
    }
}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, Dec: Decompressible<I, U>, S: Cluster<I, U, Co>>
    SquishyBall<I, U, Co, Dec, S>
{
    /// Get the unitary cost of the `SquishyBall`.
    pub const fn unitary_cost(&self) -> U {
        self.costs.unitary
    }

    /// Get the recursive cost of the `SquishyBall`.
    pub const fn recursive_cost(&self) -> U {
        self.costs.recursive
    }

    /// Gets the offset of the cluster's indices in its dataset.
    pub const fn offset(&self) -> usize {
        self.source.offset()
    }

    /// Trims the tree by removing empty children of clusters whose unitary cost
    /// is greater than the recursive cost.
    pub fn trim(&mut self, min_depth: usize) {
        if !self.children.is_empty() {
            if (self.costs.unitary <= self.costs.recursive) && (self.depth() >= min_depth) {
                self.children.clear();
            } else {
                self.children.iter_mut().for_each(|(_, _, c)| c.trim(min_depth));
            }
        }
    }

    /// Sets the costs for the tree.
    pub fn set_costs(&mut self, data: &Co) {
        self.set_unitary_cost(data);
        if self.children.is_empty() {
            self.costs.recursive = U::ZERO;
        } else {
            self.children.iter_mut().for_each(|(_, _, c)| c.set_costs(data));
            self.set_recursive_cost(data);
        }
        self.set_min_cost();
    }

    /// Calculates the unitary cost of the `Cluster`.
    fn set_unitary_cost(&mut self, data: &Co) {
        self.costs.unitary = Dataset::one_to_many(data, self.arg_center(), &self.indices().collect::<Vec<_>>())
            .into_iter()
            .map(|(_, d)| d)
            .sum();
    }

    /// Calculates the recursive cost of the `Cluster`.
    fn set_recursive_cost(&mut self, data: &Co) {
        if self.children.is_empty() {
            self.costs.recursive = U::ZERO;
        } else {
            let children = self.children.iter().map(|(_, _, c)| c.as_ref()).collect::<Vec<_>>();
            let child_costs = children.iter().map(|c| c.costs.minimum).sum::<U>();
            let child_centers = children.iter().map(|c| c.arg_center()).collect::<Vec<_>>();
            let distances = Dataset::one_to_many(data, self.arg_center(), &child_centers);

            self.costs.recursive = child_costs + distances.into_iter().map(|(_, d)| d).sum::<U>();
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

impl<
        I: Encodable + Decodable + Send + Sync,
        U: Number,
        Co: ParCompressible<I, U>,
        Dec: ParDecompressible<I, U>,
        S: ParCluster<I, U, Co>,
    > SquishyBall<I, U, Co, Dec, S>
{
    /// Sets the costs for the tree.
    pub fn par_set_costs(&mut self, data: &Co) {
        self.par_set_unitary_cost(data);
        if self.children.is_empty() {
            self.costs.recursive = U::ZERO;
        } else {
            self.children.par_iter_mut().for_each(|(_, _, c)| c.par_set_costs(data));
            self.par_set_recursive_cost(data);
        }
        self.set_min_cost();
    }

    /// Calculates the unitary cost of the `Cluster`.
    fn par_set_unitary_cost(&mut self, data: &Co) {
        self.costs.unitary = ParDataset::par_one_to_many(data, self.arg_center(), &self.indices().collect::<Vec<_>>())
            .into_iter()
            .map(|(_, d)| d)
            .sum();
    }

    /// Calculates the recursive cost of the `Cluster`.
    fn par_set_recursive_cost(&mut self, data: &Co) {
        if self.children.is_empty() {
            self.costs.recursive = U::ZERO;
        } else {
            let children = self.children.iter().map(|(_, _, c)| c.as_ref()).collect::<Vec<_>>();
            let child_costs = children.iter().map(|c| c.costs.minimum).sum::<U>();
            let child_centers = children.iter().map(|c| c.arg_center()).collect::<Vec<_>>();
            let distances = ParDataset::par_one_to_many(data, self.arg_center(), &child_centers);

            self.costs.recursive = child_costs + distances.into_iter().map(|(_, d)| d).sum::<U>();
        }
    }
}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, Dec: Decompressible<I, U>, S: Cluster<I, U, Co>>
    Adapter<I, U, Co, Dec, OffBall<I, U, Co, S>, SquishCosts<U>> for SquishyBall<I, U, Co, Dec, S>
{
    fn new_adapted(source: OffBall<I, U, Co, S>, children: Vec<(usize, U, Box<Self>)>, params: SquishCosts<U>) -> Self {
        Self {
            source,
            children,
            costs: params,
            _dc: PhantomData,
        }
    }

    fn post_traversal(&mut self) {}

    fn source(&self) -> &OffBall<I, U, Co, S> {
        &self.source
    }

    fn source_mut(&mut self) -> &mut OffBall<I, U, Co, S> {
        &mut self.source
    }

    fn take_source(self) -> OffBall<I, U, Co, S> {
        self.source
    }

    fn params(&self) -> &SquishCosts<U> {
        &self.costs
    }
}

/// Parameters for the `OffsetBall`.
#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize)]
pub struct SquishCosts<U> {
    /// Expected memory cost of recursive compression.
    recursive: U,
    /// Expected memory cost of unitary compression.
    unitary: U,
    /// The minimum expected memory cost of compression.
    minimum: U,
}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, Dec: Decompressible<I, U>, S: Cluster<I, U, Co>>
    Params<I, U, Co, Dec, S> for SquishCosts<U>
{
    fn child_params(&self, children: &[S]) -> Vec<Self> {
        children.iter().map(|_| Self::default()).collect()
    }
}

impl<
        I: Encodable + Decodable + Send + Sync,
        U: Number,
        Co: ParCompressible<I, U>,
        Dec: ParDecompressible<I, U>,
        S: ParCluster<I, U, Co> + Debug,
    > ParAdapter<I, U, Co, Dec, OffBall<I, U, Co, S>, SquishCosts<U>> for SquishyBall<I, U, Co, Dec, S>
{
    fn par_new_adapted(
        source: OffBall<I, U, Co, S>,
        children: Vec<(usize, U, Box<Self>)>,
        params: SquishCosts<U>,
    ) -> Self {
        Self::new_adapted(source, children, params)
    }

    fn par_post_traversal(&mut self) {}
}

impl<
        I: Encodable + Decodable + Send + Sync,
        U: Number,
        Co: ParCompressible<I, U>,
        Dec: ParDecompressible<I, U>,
        S: ParCluster<I, U, Co>,
    > ParParams<I, U, Co, Dec, S> for SquishCosts<U>
{
    fn par_child_params(&self, children: &[S]) -> Vec<Self> {
        Params::<I, U, Co, Dec, S>::child_params(self, children)
    }
}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, Dec: Decompressible<I, U>, S: Cluster<I, U, Co>>
    Cluster<I, U, Dec> for SquishyBall<I, U, Co, Dec, S>
{
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

    fn radius(&self) -> U {
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

    fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.source.indices()
    }

    fn set_indices(&mut self, indices: Vec<usize>) {
        self.source.set_indices(indices);
    }

    fn children(&self) -> &[(usize, U, Box<Self>)] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [(usize, U, Box<Self>)] {
        &mut self.children
    }

    fn set_children(&mut self, children: Vec<(usize, U, Box<Self>)>) {
        self.children = children;
    }

    fn take_children(&mut self) -> Vec<(usize, U, Box<Self>)> {
        std::mem::take(&mut self.children)
    }

    fn distances_to_query(&self, data: &Dec, query: &I) -> Vec<(usize, U)> {
        let leaf_bytes = data.leaf_bytes();

        let instances =
            self.leaves()
                .into_iter()
                .map(Self::offset)
                .map(|o| {
                    leaf_bytes.iter().position(|(off, _)| *off == o).unwrap_or_else(|| {
                        unreachable!("Offset not found in leaf offsets: {}, {:?}", o, data.leaf_bytes())
                    })
                })
                .map(|pos| &leaf_bytes[pos])
                .flat_map(|(o, bytes)| {
                    data.decode_leaf(bytes)
                        .into_iter()
                        .enumerate()
                        .map(|(i, p)| (i + *o, p))
                })
                .collect::<Vec<_>>();

        let instances = instances.iter().map(|(i, p)| (*i, p)).collect::<Vec<_>>();
        MetricSpace::one_to_many(data, query, &instances)
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        self.source.is_descendant_of(&other.source)
    }
}

impl<
        I: Encodable + Decodable + Send + Sync,
        U: Number,
        Co: ParCompressible<I, U>,
        Dec: ParDecompressible<I, U>,
        S: ParCluster<I, U, Co> + Debug,
    > ParCluster<I, U, Dec> for SquishyBall<I, U, Co, Dec, S>
{
    fn par_distances_to_query(&self, data: &Dec, query: &I) -> Vec<(usize, U)> {
        let leaf_bytes = data.leaf_bytes();

        let instances =
            self.leaves()
                .into_par_iter()
                .map(Self::offset)
                .map(|o| {
                    leaf_bytes.iter().position(|(off, _)| *off == o).unwrap_or_else(|| {
                        unreachable!("Offset not found in leaf offsets: {}, {:?}", o, data.leaf_bytes())
                    })
                })
                .map(|pos| &leaf_bytes[pos])
                .flat_map(|(o, bytes)| {
                    data.decode_leaf(bytes)
                        .into_par_iter()
                        .enumerate()
                        .map(|(i, p)| (i + *o, p))
                })
                .collect::<Vec<_>>();

        let instances = instances.iter().map(|(i, p)| (*i, p)).collect::<Vec<_>>();
        ParMetricSpace::par_one_to_many(data, query, &instances)
    }
}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, Dec: Decompressible<I, U>, S: Cluster<I, U, Co>>
    PartialEq for SquishyBall<I, U, Co, Dec, S>
{
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, Dec: Decompressible<I, U>, S: Cluster<I, U, Co>> Eq
    for SquishyBall<I, U, Co, Dec, S>
{
}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, Dec: Decompressible<I, U>, S: Cluster<I, U, Co>>
    PartialOrd for SquishyBall<I, U, Co, Dec, S>
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, Dec: Decompressible<I, U>, S: Cluster<I, U, Co>> Ord
    for SquishyBall<I, U, Co, Dec, S>
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.source.cmp(&other.source)
    }
}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, Dec: Decompressible<I, U>, S: Cluster<I, U, Co>>
    std::hash::Hash for SquishyBall<I, U, Co, Dec, S>
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
    }
}

#[cfg(feature = "csv")]
impl<
        I: Encodable + Decodable,
        U: Number,
        Co: Compressible<I, U>,
        Dec: Decompressible<I, U>,
        S: crate::cluster::WriteCsv<I, U, Co>,
    > crate::cluster::WriteCsv<I, U, Dec> for SquishyBall<I, U, Co, Dec, S>
{
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
