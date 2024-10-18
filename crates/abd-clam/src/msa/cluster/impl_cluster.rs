//! Implementation of the `Cluster` trait for the `PartialMSA` struct.

use distances::Number;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

use super::PartialMSA;

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D> + core::fmt::Debug> core::fmt::Debug
    for PartialMSA<I, U, D, S>
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("PartialMSA")
            .field("source", &self.source)
            .field("children", &!self.children.is_empty())
            .field("gap_ids", &self.gap_ids)
            .finish()
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> PartialEq for PartialMSA<I, U, D, S> {
    fn eq(&self, other: &Self) -> bool {
        self.source.eq(&other.source)
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Eq for PartialMSA<I, U, D, S> {}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> PartialOrd for PartialMSA<I, U, D, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Ord for PartialMSA<I, U, D, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.source.cmp(&other.source)
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> std::hash::Hash for PartialMSA<I, U, D, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Cluster<I, U, D> for PartialMSA<I, U, D, S> {
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
        core::mem::take(&mut self.children)
    }

    fn distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        self.source.distances_to_query(data, query)
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        self.source.is_descendant_of(&other.source)
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<I, U, D>> ParCluster<I, U, D>
    for PartialMSA<I, U, D, S>
{
    fn par_distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        self.source.par_distances_to_query(data, query)
    }
}
