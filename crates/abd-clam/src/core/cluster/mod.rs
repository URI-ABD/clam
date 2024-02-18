//! Provides the `Cluster` trait, which defines the most basic functionality of
//! a cluster.
//!
//! It also provides the `PartitionCriterion` trait, and implementations for
//! `PartitionCriterion` for `MaxDepth` and `MinCardinality` which are used to
//! determine when to stop partitioning the tree.

mod base;
mod children;
mod criteria;

#[allow(clippy::module_name_repetitions)]
pub use base::BaseCluster;
pub use children::Children;
pub use criteria::{MaxDepth, MinCardinality, PartitionCriteria, PartitionCriterion};

use core::{
    fmt::{Debug, Display},
    hash::Hash,
    ops::Range,
};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::{Dataset, Instance};

/// A `Cluster` represents a set of "similar" instances under some distance
/// function.
pub trait Cluster<U: Number>:
    Serialize + for<'a> Deserialize<'a> + PartialEq + Eq + PartialOrd + Ord + Debug + Hash + Display + Send + Sync
{
    /// Creates a new `Cluster` from a given dataset.
    fn new_root<I: Instance, D: Dataset<I, U>>(data: &D, seed: Option<u64>) -> Self;

    /// Recursively partitions the `Cluster` until the `PartitionCriteria` are met.
    #[must_use]
    fn partition<I, D, P>(self, data: &mut D, criteria: &P, seed: Option<u64>) -> Self
    where
        I: Instance,
        D: Dataset<I, U>,
        P: PartitionCriterion<U>;

    /// The offset of the indices of the `Cluster`'s instances in the dataset.
    fn offset(&self) -> usize;

    /// The number of points in the cluster.
    fn cardinality(&self) -> usize;

    /// The depth of the cluster in the tree.
    fn depth(&self) -> usize;

    /// The index of the instance at the `center` of the `Cluster`.
    ///
    /// TODO: Remove this method when we have "center-less" clusters.
    fn arg_center(&self) -> usize;

    /// The radius of the cluster.
    fn radius(&self) -> U;

    /// The index of the instance with the maximum distance from the `center`
    fn arg_radial(&self) -> usize;

    /// The local fractal dimension of the `å`.
    fn lfd(&self) -> f64;

    /// The two child clusters.
    fn children(&self) -> Option<[&Self; 2]>;

    /// The distance between the two poles of the `Cluster` used for partitioning.
    fn polar_distance(&self) -> Option<U>;

    /// The indices of the instances used as poles for partitioning.
    fn arg_poles(&self) -> Option<[usize; 2]>;

    /// The `name` of the `Cluster` String.
    ///
    /// This is a human-readable representation of the `Cluster`'s `offset` and
    /// `cardinality`. It is a unique identifier in the tree. It may be used to
    /// store the `Cluster` in a database, or to identify the `Cluster` in a
    /// visualization.
    fn name(&self) -> String {
        format!("{}-{}", self.offset(), self.cardinality())
    }

    /// Descends to the `Cluster` with the given `offset` and `cardinality`.
    ///
    /// If such a `Cluster` does not exist, `None` is returned.
    ///
    /// # Arguments
    ///
    /// * `offset`: The offset of the `Cluster`'s instances in the dataset.
    /// * `cardinality`: The number of instances in the `Cluster`.
    fn descend_to(&self, offset: usize, cardinality: usize) -> Option<&Self> {
        if self.offset() == offset && self.cardinality() == cardinality {
            Some(self)
        } else {
            self.children()
                .and_then(|ch| ch.iter().find_map(|c| c.descend_to(offset, cardinality)))
        }
    }

    /// Whether the `Cluster` is an ancestor of another `Cluster`.
    fn is_ancestor_of(&self, other: &Self) -> bool {
        other.depth() > self.depth()
            && self.indices().contains(&other.offset())
            && other.cardinality() < self.cardinality()
    }

    /// Whether the `Cluster` is a descendant of another `Cluster`.
    fn is_descendant_of(&self, other: &Self) -> bool {
        other.is_ancestor_of(self)
    }

    /// Whether the `Cluster` is a leaf node in the tree.
    fn is_leaf(&self) -> bool {
        self.children().is_none()
    }

    /// Whether the `Cluster` is a singleton, i.e. it contains only one instance or has a radius of zero.
    fn is_singleton(&self) -> bool {
        self.cardinality() == 1 || self.radius() == U::zero()
    }

    /// The indices of the instances in the `Cluster` after the dataset has been reordered.
    fn indices(&self) -> Range<usize> {
        self.offset()..(self.offset() + self.cardinality())
    }

    /// The subtree of the `Cluster`.
    fn subtree(&self) -> Vec<&Self> {
        let subtree = vec![self];
        match self.children() {
            Some(children) => subtree
                .into_iter()
                .chain(children.iter().flat_map(|c| c.subtree()))
                .collect(),
            None => subtree,
        }
    }

    /// The maximum depth of and leaf in the subtree of the `Cluster`.
    ///
    /// If this `Cluster` is a leaf, the maximum depth is the depth of the `Cluster`.
    fn max_leaf_depth(&self) -> usize {
        self.subtree()
            .iter()
            .map(|c| c.depth())
            .max()
            .unwrap_or_else(|| self.depth())
    }

    /// Distance from the `center` to the given instance.
    fn distance_to_instance<I: Instance, D: Dataset<I, U>>(&self, data: &D, instance: &I) -> U {
        data.query_to_one(instance, self.arg_center())
    }

    /// Distance from the `center` of this `Cluster` to the center of the
    /// `other` `Cluster`.
    fn distance_to_other<I: Instance, D: Dataset<I, U>>(&self, data: &D, other: &Self) -> U {
        data.one_to_one(self.arg_center(), other.arg_center())
    }

    /// Assuming the `Cluster` overlaps with the query ball, we return only
    /// those children that also overlap with the query ball.
    fn overlapping_children<I: Instance, D: Dataset<I, U>>(&self, data: &D, query: &I, radius: U) -> Vec<&Self> {
        if self.is_leaf() {
            Vec::new()
        } else {
            let [left, right] = self
                .children()
                .unwrap_or_else(|| unreachable!("We checked that the cluster is not a leaf."));
            let [arg_l, arg_r] = self
                .arg_poles()
                .unwrap_or_else(|| unreachable!("We checked that the cluster is not a leaf."));
            let polar_distance = self
                .polar_distance()
                .unwrap_or_else(|| unreachable!("We checked that the cluster is not a leaf."));

            let ql = data.query_to_one(query, arg_l);
            let qr = data.query_to_one(query, arg_r);

            let swap = ql < qr;
            let (ql, qr) = if swap { (qr, ql) } else { (ql, qr) };

            if (ql + qr) * (ql - qr) <= U::from(2) * polar_distance * radius {
                vec![left, right]
            } else if swap {
                vec![left]
            } else {
                vec![right]
            }
        }
    }

    /// Saves a `Cluster` to a given location.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to the `Cluster` file.
    ///
    /// # Errors
    ///
    /// * If the file cannot be created.
    /// * If the file cannot be serialized.
    fn save(&self, path: &Path) -> Result<(), String> {
        let mut writer = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
        bincode::serialize_into(&mut writer, self).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Loads a `Cluster` from a given location.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to the `Cluster` file.
    ///
    /// # Returns
    ///
    /// * The `Cluster` loaded from the file.
    ///
    /// # Errors
    ///
    /// * If the file cannot be opened.
    /// * If the file cannot be deserialized.
    fn load(path: &Path) -> Result<Self, String> {
        let reader = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
        bincode::deserialize_from(reader).map_err(|e| e.to_string())
    }
}
