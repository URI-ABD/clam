//! An adaptation of `Ball` that allows for compression of the dataset and
//! search in the compressed space.

use core::fmt::Debug;

use distances::Number;
use rayon::prelude::*;

use crate::{
    adapter::{Adapter, ParAdapter, ParParams, Params},
    cakes::OffBall,
    cluster::ParCluster,
    dataset::ParDataset,
    Cluster, Dataset, MetricSpace,
};

use super::{Decodable, Decompressible};

/// A variant of `Ball` that stores indices after reordering the dataset.
pub struct SquishyBall<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D>> {
    /// The `Cluster` type that the `OffsetBall` is based on.
    source: OffBall<I, U, D, S>,
    /// The children of the `Cluster`.
    children: Vec<(usize, U, Box<Self>)>,
    /// Parameters for the `OffsetBall`.
    costs: SquishCosts<U>,
}

impl<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D> + Debug> Debug for SquishyBall<I, U, D, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SquishyBall")
            .field("source", &self.source)
            .field("children", &self.children.is_empty())
            .field("recursive_cost", &self.costs.recursive)
            .field("unitary_cost", &self.costs.unitary)
            .field("minimum_cost", &self.costs.minimum)
            .finish()
    }
}

impl<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D>> SquishyBall<I, U, D, S> {
    /// Creates a new `SquishyBall` tree from a `Cluster`.
    ///
    /// # Arguments
    ///
    /// - `source`: The `Cluster` to adapt into a `SquishyBall`.
    /// - `data`: The dataset containing the instances.
    /// - `trim`: Whether to trim the tree after creating it, i.e. remove
    ///   children of clusters whose unitary cost of compression is greater than
    ///   the recursive cost.
    pub fn from_root(source: OffBall<I, U, D, S>, data: &D, trim: bool) -> Self {
        let (mut root, _) = Self::adapt(source, None);
        root.set_costs(data);
        if trim {
            root.trim();
        }
        root
    }

    /// Trims the tree by removing empty children of clusters whose unitary cost
    /// is greater than the recursive cost.
    fn trim(&mut self) {
        if !self.children.is_empty() {
            if self.costs.unitary <= self.costs.recursive {
                self.children.clear();
            } else {
                self.children.iter_mut().for_each(|(_, _, c)| c.trim());
            }
        }
    }

    /// Sets the costs for the tree.
    fn set_costs(&mut self, data: &D) {
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
    fn set_unitary_cost(&mut self, data: &D) {
        self.costs.unitary = Dataset::one_to_many(data, self.arg_center(), &self.indices().collect::<Vec<_>>())
            .into_iter()
            .map(|(_, d)| d)
            .sum();
    }

    /// Calculates the recursive cost of the `Cluster`.
    fn set_recursive_cost(&mut self, data: &D) {
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

    /// Gets the offset of the cluster's indices in its dataset.
    pub const fn offset(&self) -> usize {
        self.source.offset()
    }
}

impl<I: Decodable + Send + Sync, U: Number, D: Decompressible<I, U> + ParDataset<I, U>, S: ParCluster<I, U, D>>
    SquishyBall<I, U, D, S>
{
    /// Creates a new `SquishyBall` tree from a `Cluster`.
    ///
    /// # Arguments
    ///
    /// - `source`: The `Cluster` to adapt into a `SquishyBall`.
    /// - `data`: The dataset containing the instances.
    /// - `trim`: Whether to trim the tree after creating it, i.e. remove
    ///   children of clusters whose unitary cost of compression is greater than
    ///   the recursive cost.
    pub fn par_from_root(source: OffBall<I, U, D, S>, data: &D, trim: bool) -> Self {
        let (mut root, _) = Self::par_adapt(source, None);
        root.par_set_costs(data);
        if trim {
            root.trim();
        }
        root
    }

    /// Sets the costs for the tree.
    fn par_set_costs(&mut self, data: &D) {
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
    fn par_set_unitary_cost(&mut self, data: &D) {
        self.costs.unitary = ParDataset::par_one_to_many(data, self.arg_center(), &self.indices().collect::<Vec<_>>())
            .into_iter()
            .map(|(_, d)| d)
            .sum();
    }

    /// Calculates the recursive cost of the `Cluster`.
    fn par_set_recursive_cost(&mut self, data: &D) {
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

impl<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D>>
    Adapter<I, U, D, OffBall<I, U, D, S>, SquishCosts<U>> for SquishyBall<I, U, D, S>
{
    fn adapt(source: OffBall<I, U, D, S>, params: Option<SquishCosts<U>>) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
        let (mut source, indices, children) = source.disassemble();
        source.set_indices(indices.clone());
        let params = params.unwrap_or_default();

        let cluster = if children.is_empty() {
            Self::newly_adapted(source, Vec::new(), params)
        } else {
            let (arg_extrema, others) = children
                .into_iter()
                .map(|(a, b, c)| (a, (b, c)))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let (extents, children) = others.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
            let (children, _) = params
                .child_params(&children)
                .into_iter()
                .zip(children)
                .map(|(p, c)| Self::adapt(*c, Some(p)))
                .unzip::<_, _, Vec<_>, Vec<_>>();

            let children = arg_extrema
                .into_iter()
                .zip(extents)
                .zip(children)
                .map(|((a, b), c)| (a, b, Box::new(c)))
                .collect();

            Self::newly_adapted(source, children, params)
        };

        (cluster, indices)
    }

    fn newly_adapted(source: OffBall<I, U, D, S>, children: Vec<(usize, U, Box<Self>)>, params: SquishCosts<U>) -> Self {
        Self {
            source,
            children,
            costs: params,
        }
    }

    fn source(&self) -> &OffBall<I, U, D, S> {
        &self.source
    }

    fn source_mut(&mut self) -> &mut OffBall<I, U, D, S> {
        &mut self.source
    }
}

/// Parameters for the `OffsetBall`.
#[derive(Debug, Default, Copy, Clone)]
struct SquishCosts<U> {
    /// Expected memory cost of recursive compression.
    recursive: U,
    /// Expected memory cost of unitary compression.
    unitary: U,
    /// The minimum expected memory cost of compression.
    minimum: U,
}

impl<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D>> Params<I, U, D, S> for SquishCosts<U> {
    fn child_params<B: AsRef<S>>(&self, children: &[B]) -> Vec<Self> {
        children.iter().map(|_| Self::default()).collect()
    }
}

impl<I: Decodable + Send + Sync, U: Number, D: Decompressible<I, U> + ParDataset<I, U>, S: ParCluster<I, U, D>>
    ParAdapter<I, U, D, OffBall<I, U, D, S>, SquishCosts<U>> for SquishyBall<I, U, D, S>
{
    fn par_adapt(source: OffBall<I, U, D, S>, params: Option<SquishCosts<U>>) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
        let (mut source, indices, children) = source.disassemble();
        source.set_indices(indices.clone());
        let params = params.unwrap_or_default();

        let cluster = if children.is_empty() {
            Self::newly_adapted(source, Vec::new(), params)
        } else {
            let (arg_extrema, others) = children
                .into_par_iter()
                .map(|(a, b, c)| (a, (b, c)))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let (extents, children) = others.into_par_iter().unzip::<_, _, Vec<_>, Vec<_>>();
            let (children, _) = params
                .child_params(&children)
                .into_par_iter()
                .zip(children.into_par_iter())
                .map(|(p, c)| Self::par_adapt(*c, Some(p)))
                .unzip::<_, _, Vec<_>, Vec<_>>();

            let children = arg_extrema
                .into_iter()
                .zip(extents)
                .zip(children)
                .map(|((a, b), c)| (a, b, Box::new(c)))
                .collect();

            Self::newly_adapted(source, children, params)
        };

        (cluster, indices)
    }
}

impl<I: Decodable + Send + Sync, U: Number, D: Decompressible<I, U> + ParDataset<I, U>, S: ParCluster<I, U, D>>
    ParParams<I, U, D, S> for SquishCosts<U>
{
    fn par_child_params<B: AsRef<S>>(&self, children: &[B]) -> Vec<Self> {
        self.child_params(children)
    }
}

impl<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D>> Cluster<I, U, D>
    for SquishyBall<I, U, D, S>
{
    fn new(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> (Self, usize) {
        let (source, arg_radial) = OffBall::<I, U, D, S>::new(data, indices, depth, seed);
        let ball = Self {
            source,
            children: Vec::new(),
            costs: SquishCosts::default(),
        };
        (ball, arg_radial)
    }

    fn disassemble(self) -> (Self, Vec<usize>, Vec<(usize, U, Box<Self>)>) {
        let indices = self.indices().collect();
        let Self {
            source,
            children,
            costs: params,
        } = self;
        (
            Self {
                source,
                children: Vec::new(),
                costs: params,
            },
            indices,
            children,
        )
    }

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

    fn set_children(self, children: Vec<(usize, U, Self)>) -> Self {
        let children = children.into_iter().map(|(i, r, c)| (i, r, Box::new(c))).collect();
        Self { children, ..self }
    }

    fn find_extrema(&self, data: &D) -> Vec<usize> {
        self.source.find_extrema(data)
    }

    fn distances(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        self.leaves()
            .into_iter()
            .map(Self::offset)
            .flat_map(|o| data.decode_leaf(o))
            .zip(self.indices())
            .map(|(p, i)| (i, MetricSpace::one_to_one(data, query, &p)))
            .collect()
    }
}

impl<I: Decodable + Send + Sync, U: Number, D: Decompressible<I, U> + ParDataset<I, U>, S: ParCluster<I, U, D>>
    ParCluster<I, U, D> for SquishyBall<I, U, D, S>
{
    fn par_new(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> (Self, usize) {
        let (source, arg_radial) = OffBall::<I, U, D, S>::par_new(data, indices, depth, seed);
        let ball = Self {
            source,
            children: Vec::new(),
            costs: SquishCosts::default(),
        };
        (ball, arg_radial)
    }

    fn par_find_extrema(&self, data: &D) -> Vec<usize> {
        self.source.par_find_extrema(data)
    }
}

impl<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D>> PartialEq for SquishyBall<I, U, D, S> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D>> Eq for SquishyBall<I, U, D, S> {}

impl<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D>> PartialOrd for SquishyBall<I, U, D, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D>> Ord for SquishyBall<I, U, D, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.source.cmp(&other.source)
    }
}

impl<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D>> std::hash::Hash for SquishyBall<I, U, D, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
    }
}
