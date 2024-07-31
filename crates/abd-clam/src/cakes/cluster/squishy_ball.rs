//! An adaptation of `Ball` that allows for compression of the dataset and
//! search in the compressed space.

use core::fmt::Debug;

use distances::Number;
use rayon::prelude::*;

use crate::{
    adapter::{Adapter, ParAdapter, ParParams, Params},
    cluster::ParCluster,
    dataset::ParDataset,
    Cluster, Dataset,
};

/// A variant of `Ball` that stores indices after reordering the dataset.
pub struct SquishyBall<U: Number, S: Cluster<U>> {
    /// The `Cluster` type that the `OffsetBall` is based on.
    source: S,
    /// The children of the `Cluster`.
    children: Vec<(usize, U, Box<Self>)>,
    /// Parameters for the `OffsetBall`.
    params: SquishCosts<U>,
}

impl<U: Number, S: Cluster<U> + Debug> Debug for SquishyBall<U, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SquishyBall")
            .field("source", &self.source)
            .field("children", &self.children.is_empty())
            .field("params", &self.params)
            .finish()
    }
}

impl<U: Number, S: Cluster<U>> SquishyBall<U, S> {
    /// Creates a new `SquishyBall` tree from a `Cluster`.
    ///
    /// # Arguments
    ///
    /// - `source`: The `Cluster` to adapt into a `SquishyBall`.
    /// - `data`: The dataset containing the instances.
    /// - `trim`: Whether to trim the tree after creating it, i.e. remove
    /// children of clusters whose unitary cost of compression is greater than
    /// the recursive cost.
    pub fn from_root<I, D: Dataset<I, U>>(source: S, data: &D, trim: bool) -> Self {
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
            if self.params.unitary <= self.params.recursive {
                self.children.clear();
            } else {
                self.children.iter_mut().for_each(|(_, _, c)| c.trim());
            }
        }
    }

    /// Sets the costs for the tree.
    fn set_costs<I, D: Dataset<I, U>>(&mut self, data: &D) {
        self.set_unitary_cost(data);
        if self.children.is_empty() {
            self.params.recursive = U::ZERO;
        } else {
            self.children.iter_mut().for_each(|(_, _, c)| c.set_costs(data));
            self.set_recursive_cost(data);
        }
        self.set_min_cost();
    }

    /// Calculates the unitary cost of the `Cluster`.
    fn set_unitary_cost<I, D: Dataset<I, U>>(&mut self, data: &D) {
        self.params.unitary = Dataset::one_to_many(data, self.arg_center(), &self.indices().collect::<Vec<_>>())
            .into_iter()
            .map(|(_, d)| d)
            .sum();
    }

    /// Calculates the recursive cost of the `Cluster`.
    fn set_recursive_cost<I, D: Dataset<I, U>>(&mut self, data: &D) {
        if self.children.is_empty() {
            self.params.recursive = U::ZERO;
        } else {
            let children = self.children.iter().map(|(_, _, c)| c.as_ref()).collect::<Vec<_>>();
            let child_costs = children.iter().map(|c| c.params.minimum).sum::<U>();
            let child_centers = children.iter().map(|c| c.arg_center()).collect::<Vec<_>>();
            let distances = Dataset::one_to_many(data, self.arg_center(), &child_centers);

            self.params.recursive = child_costs + distances.into_iter().map(|(_, d)| d).sum::<U>();
        }
    }

    /// Sets the minimum cost of the `Cluster`.
    fn set_min_cost(&mut self) {
        self.params.minimum = if self.params.recursive < self.params.unitary {
            self.params.recursive
        } else {
            self.params.unitary
        };
    }
}

impl<U: Number, S: ParCluster<U>> SquishyBall<U, S> {
    /// Creates a new `SquishyBall` tree from a `Cluster`.
    ///
    /// # Arguments
    ///
    /// - `source`: The `Cluster` to adapt into a `SquishyBall`.
    /// - `data`: The dataset containing the instances.
    /// - `trim`: Whether to trim the tree after creating it, i.e. remove
    /// children of clusters whose unitary cost of compression is greater than
    /// the recursive cost.
    pub fn par_from_root<I: Send + Sync, D: ParDataset<I, U>>(source: S, data: &D, trim: bool) -> Self {
        let (mut root, _) = Self::par_adapt(source, None);
        root.par_set_costs(data);
        if trim {
            root.trim();
        }
        root
    }

    /// Sets the costs for the tree.
    fn par_set_costs<I: Send + Sync, D: ParDataset<I, U>>(&mut self, data: &D) {
        self.par_set_unitary_cost(data);
        if self.children.is_empty() {
            self.params.recursive = U::ZERO;
        } else {
            self.children.par_iter_mut().for_each(|(_, _, c)| c.par_set_costs(data));
            self.par_set_recursive_cost(data);
        }
        self.set_min_cost();
    }

    /// Calculates the unitary cost of the `Cluster`.
    fn par_set_unitary_cost<I: Send + Sync, D: ParDataset<I, U>>(&mut self, data: &D) {
        self.params.unitary = ParDataset::par_one_to_many(data, self.arg_center(), &self.indices().collect::<Vec<_>>())
            .into_iter()
            .map(|(_, d)| d)
            .sum();
    }

    /// Calculates the recursive cost of the `Cluster`.
    fn par_set_recursive_cost<I: Send + Sync, D: ParDataset<I, U>>(&mut self, data: &D) {
        if self.children.is_empty() {
            self.params.recursive = U::ZERO;
        } else {
            let children = self.children.iter().map(|(_, _, c)| c.as_ref()).collect::<Vec<_>>();
            let child_costs = children.iter().map(|c| c.params.minimum).sum::<U>();
            let child_centers = children.iter().map(|c| c.arg_center()).collect::<Vec<_>>();
            let distances = ParDataset::par_one_to_many(data, self.arg_center(), &child_centers);

            self.params.recursive = child_costs + distances.into_iter().map(|(_, d)| d).sum::<U>();
        }
    }
}

impl<U: Number, S: Cluster<U>> Adapter<U, S, SquishCosts<U>> for SquishyBall<U, S> {
    fn adapt(source: S, params: Option<SquishCosts<U>>) -> (Self, Vec<usize>)
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

    fn newly_adapted(source: S, children: Vec<(usize, U, Box<Self>)>, params: SquishCosts<U>) -> Self {
        Self {
            source,
            children,
            params,
        }
    }

    fn source(&self) -> &S {
        &self.source
    }

    fn source_mut(&mut self) -> &mut S {
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

impl<U: Number, S: Cluster<U>> Params<U, S> for SquishCosts<U> {
    fn child_params<B: AsRef<S>>(&self, children: &[B]) -> Vec<Self> {
        children.iter().map(|_| Self::default()).collect()
    }
}

impl<U: Number, S: ParCluster<U>> ParAdapter<U, S, SquishCosts<U>> for SquishyBall<U, S> {
    fn par_adapt(source: S, params: Option<SquishCosts<U>>) -> (Self, Vec<usize>)
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

impl<U: Number, S: ParCluster<U>> ParParams<U, S> for SquishCosts<U> {
    fn par_child_params<B: AsRef<S>>(&self, children: &[B]) -> Vec<Self> {
        self.child_params(children)
    }
}

impl<U: Number, S: Cluster<U>> Cluster<U> for SquishyBall<U, S> {
    fn new<I, D: Dataset<I, U>>(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> (Self, usize) {
        let (source, arg_radial) = S::new(data, indices, depth, seed);
        let ball = Self {
            source,
            children: Vec::new(),
            params: SquishCosts::default(),
        };
        (ball, arg_radial)
    }

    fn disassemble(self) -> (Self, Vec<usize>, Vec<(usize, U, Box<Self>)>) {
        let indices = self.indices().collect();
        let Self {
            source,
            children,
            params,
        } = self;
        (
            Self {
                source,
                children: Vec::new(),
                params,
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

    fn find_extrema<I, D: Dataset<I, U>>(&self, data: &D) -> Vec<usize> {
        self.source.find_extrema(data)
    }
}

impl<U: Number, S: ParCluster<U>> ParCluster<U> for SquishyBall<U, S> {
    fn par_new<I: Send + Sync, D: ParDataset<I, U>>(
        data: &D,
        indices: &[usize],
        depth: usize,
        seed: Option<u64>,
    ) -> (Self, usize) {
        let (source, arg_radial) = S::par_new(data, indices, depth, seed);
        let ball = Self {
            source,
            children: Vec::new(),
            params: SquishCosts::default(),
        };
        (ball, arg_radial)
    }

    fn par_find_extrema<I: Send + Sync, D: ParDataset<I, U>>(&self, data: &D) -> Vec<usize> {
        self.source.par_find_extrema(data)
    }
}

impl<U: Number, S: Cluster<U>> PartialEq for SquishyBall<U, S> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl<U: Number, S: Cluster<U>> Eq for SquishyBall<U, S> {}

impl<U: Number, S: Cluster<U>> PartialOrd for SquishyBall<U, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: Number, S: Cluster<U>> Ord for SquishyBall<U, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.source.cmp(&other.source)
    }
}

impl<U: Number, S: Cluster<U>> std::hash::Hash for SquishyBall<U, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
    }
}
