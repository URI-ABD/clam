//! The `Children` of a `Cluster`.

use distances::Number;
use rayon::prelude::*;

use super::{
    adapter::{Adapter, ParAdapter, ParParams, Params},
    Ball,
};

/// The `Children` of a `Cluster`.
#[derive(Debug, Clone)]
pub struct Children<U: Number, C> {
    /// The children of the `Cluster`.
    clusters: Vec<Box<C>>,
    /// The indices of the extremal points used to partition the `Cluster`.
    arg_extrema: Vec<usize>,
    /// The pairwise distances between the extrema.
    extremal_distances: Vec<Vec<U>>,
}

impl<U: Number> Children<U, Ball<U>> {
    /// Adapt the `Children` of a `Ball` into another `Cluster` type.
    pub fn adapt<C: Adapter<U, P>, P: Params<U>>(self, params: &P) -> (Children<U, C>, Vec<usize>) {
        let child_params = params.child_params(&self.clusters);
        let (clusters, indices) = self
            .clusters
            .into_iter()
            .zip(child_params)
            .map(|(child, params)| C::adapt(*child, Some(params)))
            .map(|(c, i)| (Box::new(c), i))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let children = Children {
            clusters,
            arg_extrema: self.arg_extrema,
            extremal_distances: self.extremal_distances,
        };
        let indices = indices.into_iter().flatten().collect();
        (children, indices)
    }

    /// Parallel version of the `adapt` method.
    pub fn par_adapt<C: ParAdapter<U, P>, P: ParParams<U>>(self, params: &P) -> (Children<U, C>, Vec<usize>) {
        let child_params = params.par_child_params(&self.clusters);
        let (clusters, indices) = self
            .clusters
            .into_par_iter()
            .zip(child_params.into_par_iter())
            .map(|(child, params)| C::par_adapt(*child, Some(params)))
            .map(|(c, i)| (Box::new(c), i))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let children = Children {
            clusters,
            arg_extrema: self.arg_extrema,
            extremal_distances: self.extremal_distances,
        };
        let indices = indices.into_iter().flatten().collect();
        (children, indices)
    }
}

impl<U: Number, C> Children<U, C> {
    /// Creates a new `Children`.
    ///
    /// # Arguments
    ///
    /// - `children`: The children of the `Cluster`.
    /// - `arg_extrema`: The indices of the extremal points used to partition the `Cluster`.
    pub fn new(clusters: Vec<C>, arg_extrema: Vec<usize>, extremal_distances: Vec<Vec<U>>) -> Self {
        Self {
            clusters: clusters.into_iter().map(Box::new).collect(),
            arg_extrema,
            extremal_distances,
        }
    }

    /// Returns the children of the `Cluster`.
    #[must_use]
    pub fn clusters(&self) -> Vec<&C> {
        self.clusters.iter().map(AsRef::as_ref).collect::<Vec<_>>()
    }

    /// Returns the children of the `Cluster` as mutable references.
    #[must_use]
    pub fn clusters_mut(&mut self) -> Vec<&mut C> {
        self.clusters.iter_mut().map(AsMut::as_mut).collect::<Vec<_>>()
    }

    /// Returns the indices of the extremal points used to partition the `Cluster`.
    #[must_use]
    pub fn arg_extrema(&self) -> &[usize] {
        &self.arg_extrema
    }

    /// Returns the pairwise distances between the extrema.
    #[must_use]
    pub fn extremal_distances(&self) -> &[Vec<U>] {
        &self.extremal_distances
    }
}
