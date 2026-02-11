//! Getters for `Cluster` properties.

use crate::DistanceValue;

use super::Cluster;

/// Getters for `Cluster` properties.
impl<T, A> Cluster<T, A> {
    /// Returns the depth of this cluster in the tree.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Gets the center item index.
    pub const fn center_index(&self) -> usize {
        self.center_index
    }

    /// Gets the cardinality of this cluster.
    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Gets the radius of this cluster.
    pub const fn radius(&self) -> T
    where
        T: DistanceValue,
    {
        self.radius
    }

    /// Gets the local fractal dimension of this cluster.
    pub const fn lfd(&self) -> f64 {
        self.lfd
    }

    /// If the cluster was partitioned, gets the center indices of the centers of the children and the span of this cluster.
    pub fn child_center_indices_and_span(&self) -> Option<(&[usize], T)>
    where
        T: DistanceValue,
    {
        self.children
            .as_ref()
            .map(|(child_center_indices, span)| (child_center_indices.as_ref(), *span))
    }

    /// If the cluster was partitioned, gets the center indices of the children of this cluster.
    pub fn child_center_indices(&self) -> Option<&[usize]> {
        self.children.as_ref().map(|(child_center_indices, _)| child_center_indices.as_ref())
    }

    /// If the cluster was partitioned, gets the span of this cluster.
    pub fn span(&self) -> Option<T>
    where
        T: DistanceValue,
    {
        self.children.as_ref().map(|(_, span)| *span)
    }

    /// Returns the index of the center of the parent cluster. For the root cluster, this will be identical to `center_index`.
    pub const fn parent_center_index(&self) -> Option<usize> {
        self.parent_center_index
    }

    /// Returns a reference to the cluster's annotation.
    pub const fn annotation(&self) -> &A {
        &self.annotation
    }

    /// Returns a mutable reference to the cluster's annotation.
    pub const fn annotation_mut(&mut self) -> &mut A {
        &mut self.annotation
    }

    /// Removes and returns the cluster's annotation, leaving the default value in its place.
    pub fn take_annotation(&mut self) -> A
    where
        A: Default,
    {
        core::mem::take(&mut self.annotation)
    }

    /// Clears the annotation of this cluster and allows specifying a new annotation type.
    pub fn without_annotation<B>(&self) -> Cluster<T, B>
    where
        T: Clone,
        B: Default,
    {
        Cluster {
            depth: self.depth,
            center_index: self.center_index,
            cardinality: self.cardinality,
            radius: self.radius.clone(),
            lfd: self.lfd,
            children: self.children.clone(),
            annotation: B::default(),
            parent_center_index: self.parent_center_index,
        }
    }

    /// Returns true if this cluster is a singleton (i.e., contains exactly one item or has a radius of zero).
    pub fn is_singleton(&self) -> bool
    where
        T: DistanceValue,
    {
        self.cardinality == 1 || self.radius.is_zero()
    }

    /// Returns true if this cluster is a leaf (i.e., has no children).
    pub const fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Returns a `Range` that can be used to index into the `items` array of the `Tree` for all items in this cluster, including the center item.
    pub const fn items_indices(&self) -> std::ops::Range<usize> {
        self.center_index..(self.center_index + self.cardinality)
    }

    /// Returns a `Range` that can be used to index into the `items` array of the `Tree` for all items in this cluster, excluding the center item.
    pub const fn subtree_indices(&self) -> std::ops::Range<usize> {
        (self.center_index + 1)..(self.center_index + self.cardinality)
    }
}
