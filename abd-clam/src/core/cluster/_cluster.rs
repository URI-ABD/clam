//! The `Cluster` is the heart of CLAM. It provides the ability to perform a
//! divisive hierarchical cluster of arbitrary datasets in arbitrary metric
//! spaces.

use core::{
    cmp::Ordering,
    fmt::{Display, Formatter},
    hash::{Hash, Hasher},
    ops::Range,
};
use distances::Number;
use serde::{Deserialize, Serialize};
use std::fmt::Write;

use crate::{utils, Dataset, Instance, PartitionCriteria, PartitionCriterion};

/// Ratios are used for anomaly detection and related applications.
use crate::core::cluster::Ratios;

/// A `Cluster` represents a collection of "similar" instances from a metric-`Space`.
///
/// `Cluster`s can be unwieldy to use directly unless one has a good grasp of
/// the underlying invariants. We anticipate that most users' needs will be well
/// met by the higher-level abstractions, e.g. `Tree`, `Graph`, `CAKES`, etc.
///
/// For now, `Cluster` names are unique within a single tree. We plan on adding
/// tree-based prefixes which will make names unique across multiple trees.
#[derive(Debug)]
pub struct Cluster<U: Number> {
    /// The `Cluster`'s history in the tree.
    history: Vec<bool>,
    /// The seed used in the random number generator for this `Cluster`.
    seed: Option<u64>,
    /// The offset of the indices of the `Cluster`'s instances in the dataset.
    offset: usize,
    /// The number of instances in the `Cluster`.
    cardinality: usize,
    /// The index of the `center` instance in the dataset.
    arg_center: usize,
    /// The index of the `radial` instance in the dataset.
    arg_radial: usize,
    /// The distance from the `center` to the `radial` instance.
    radius: U,
    /// The local fractal dimension of the `Cluster`.
    lfd: f64,
    /// The children of the `Cluster`.
    pub(crate) children: Option<Children<U>>,
    /// The six `Cluster` ratios used for anomaly detection and related applications.
    #[allow(dead_code)]
    ratios: Option<Ratios>,
}

/// The children of a `Cluster`.
#[derive(Debug)]
pub struct Children<U: Number> {
    /// The left child of the `Cluster`.
    pub(crate) left: Box<Cluster<U>>,
    /// The right child of the `Cluster`.
    pub(crate) right: Box<Cluster<U>>,
    /// The left pole of the `Cluster` (i.e. the instance used to identify
    /// instances for the left child).
    pub(crate) arg_l: usize,
    /// The right pole of the `Cluster` (i.e. the instance used to identify
    /// instances for the right child).
    pub(crate) arg_r: usize,
    /// The distance from the `l_pole` to the `r_pole` instance.
    pub(crate) polar_distance: U,
}

impl<U: Number> PartialEq for Cluster<U> {
    fn eq(&self, other: &Self) -> bool {
        self.history == other.history
    }
}

impl<U: Number> Eq for Cluster<U> {}

impl<U: Number> PartialOrd for Cluster<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: Number> Ord for Cluster<U> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.depth().cmp(&other.depth()) {
            Ordering::Equal => self.offset.cmp(&other.offset),
            ordering => ordering,
        }
    }
}

impl<U: Number> Hash for Cluster<U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.offset, self.cardinality).hash(state);
    }
}

impl<U: Number> Display for Cluster<U> {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl<U: Number> Cluster<U> {
    /// The number of instances in the `Cluster`.
    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// The index of the instance at the `center` of the `Cluster`.
    pub const fn arg_center(&self) -> usize {
        self.arg_center
    }

    /// The index of the instance with the maximum distance from the `center`
    pub const fn arg_radial(&self) -> usize {
        self.arg_radial
    }

    /// The distance from the `center` to the `radial` instance.
    pub const fn radius(&self) -> U {
        self.radius
    }

    /// The local fractal dimension of the `Cluster`.
    pub const fn lfd(&self) -> f64 {
        self.lfd
    }

    /// The six `Cluster` ratios used for anomaly detection and related
    /// applications.
    ///
    /// These ratios are:
    ///
    /// * child-cardinality / parent-cardinality.
    /// * child-radius / parent-radius.
    /// * child-lfd / parent-lfd.
    /// * EMA of child-cardinality / parent-cardinality.
    /// * EMA of child-radius / parent-radius.
    /// * EMA of child-lfd / parent-lfd.
    pub const fn ratios(&self) -> Option<Ratios> {
        self.ratios
    }

    /// Creates a new root `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `data`: on which to create the `Cluster`.
    /// * `indices`: The indices of instances from the `dataset` that are contained in the `Cluster`.
    /// * `seed`: The seed used in the random number generator for this `Cluster`.
    pub fn new_root<I: Instance, D: Dataset<I, U>>(data: &D, seed: Option<u64>) -> Self {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        Self::new(data, seed, vec![true], 0, &indices)
    }

    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `data`: on which to create the `Cluster`.
    /// * `seed`: The seed used in the random number generator for this `Cluster`.
    /// * `history`: The `Cluster`'s history in the tree.
    /// * `offset`: The offset of the indices of the `Cluster`'s instances in the dataset.
    /// * `indices`: The indices of instances from the `dataset` that are contained in the `Cluster`.
    fn new<I: Instance, D: Dataset<I, U>>(
        data: &D,
        seed: Option<u64>,
        history: Vec<bool>,
        offset: usize,
        indices: &[usize],
    ) -> Self {
        let cardinality = indices.len();

        // TODO: Explore with different values for the threshold e.g. 10, 100, 1000, etc.
        let arg_samples = if cardinality < 100 {
            indices.to_vec()
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let n = (indices.len().as_f64().sqrt()) as usize;
            data.choose_unique(n, indices, seed)
        };

        let Some(arg_center) = data.median(&arg_samples) else {
            unreachable!("The cluster should have at least one instance.")
        };

        let center_distances = data.one_to_many(arg_center, indices);
        let Some((arg_radial, radius)) = utils::arg_max(&center_distances) else {
            unreachable!("The cluster should have at least one instance.")
        };
        let arg_radial = indices[arg_radial];

        let lfd = utils::compute_lfd(radius, &center_distances);

        Self {
            history,
            seed,
            offset,
            cardinality,
            arg_center,
            arg_radial,
            radius,
            lfd,
            children: None,
            ratios: None,
        }
    }

    /// Partitions the `Cluster` into two children if the `Cluster` meets the
    /// given `PartitionCriteria`.
    ///
    /// This method should only be called on a root `Cluster`. It is user error
    /// to call this method on a non-root `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `data`: The `Dataset` for the `Cluster`.
    /// * `criteria`: The `PartitionCriteria` to use for partitioning.
    ///
    /// # Returns
    ///
    /// * The `Cluster` on which the method was called after partitioning
    /// recursively until the `PartitionCriteria` is no longer met on any of the
    /// leaf `Cluster`s.
    #[must_use]
    pub fn partition<I: Instance, D: Dataset<I, U>>(mut self, data: &mut D, criteria: &PartitionCriteria<U>) -> Self {
        let mut indices = (0..self.cardinality).collect::<Vec<_>>();
        (self, indices) = self._partition(data, criteria, indices);
        data.permute_instances(&indices)
            .unwrap_or_else(|_| unreachable!("All indices are valid."));

        self
    }

    /// Recursive helper function for `partition`.
    fn _partition<I: Instance, D: Dataset<I, U>>(
        mut self,
        data: &D,
        criteria: &PartitionCriteria<U>,
        mut indices: Vec<usize>,
    ) -> (Self, Vec<usize>) {
        if criteria.check(&self) {
            let ([(arg_l, l_indices), (arg_r, r_indices)], polar_distance) = self.partition_once(data, indices);

            let r_offset = self.offset + l_indices.len();

            let (l_history, r_history) = (self.child_history(false), self.child_history(true));

            let ((left, l_indices), (right, r_indices)) = rayon::join(
                || Self::new(data, self.seed, l_history, self.offset, &l_indices)._partition(data, criteria, l_indices),
                || Self::new(data, self.seed, r_history, r_offset, &r_indices)._partition(data, criteria, r_indices),
            );

            let arg_l = utils::pos_val(&l_indices, arg_l)
                .map_or_else(|| unreachable!("We know the left pole is in the indices."), |(i, _)| i);
            let arg_r = utils::pos_val(&r_indices, arg_r)
                .map_or_else(|| unreachable!("We know the right pole is in the indices."), |(i, _)| i);

            self.children = Some(Children {
                left: Box::new(left),
                right: Box::new(right),
                arg_l: self.offset + arg_l,
                arg_r: r_offset + arg_r,
                polar_distance,
            });

            indices = l_indices.into_iter().chain(r_indices).collect::<Vec<_>>();
        }

        // reset the indices to center and radial indices for data reordering
        let arg_center = utils::pos_val(&indices, self.arg_center)
            .map_or_else(|| unreachable!("We know the center is in the indices."), |(i, _)| i);
        self.arg_center = self.offset + arg_center;

        let arg_radial = utils::pos_val(&indices, self.arg_radial)
            .map_or_else(|| unreachable!("We know the radial is in the indices."), |(i, _)| i);
        self.arg_radial = self.offset + arg_radial;

        (self, indices)
    }

    /// Partitions the `Cluster` into two children once.
    fn partition_once<I: Instance, D: Dataset<I, U>>(
        &self,
        data: &D,
        indices: Vec<usize>,
    ) -> ([(usize, Vec<usize>); 2], U) {
        let l_distances = data.one_to_many(self.arg_radial, &indices);

        let Some((arg_r, polar_distance)) = utils::arg_max(&l_distances) else {
            unreachable!("The cluster should have at least one instance.")
        };
        let arg_r = indices[arg_r];
        let r_distances = data.one_to_many(arg_r, &indices);

        let (l_indices, r_indices) = indices
            .into_iter()
            .zip(l_distances)
            .zip(r_distances)
            .filter(|&((i, _), _)| i != self.arg_radial && i != arg_r)
            .partition::<Vec<_>, _>(|&((_, l), r)| l <= r);

        let (l_indices, r_indices) = {
            let mut l_indices = Self::drop_distances(l_indices);
            let mut r_indices = Self::drop_distances(r_indices);

            l_indices.push(self.arg_radial);
            r_indices.push(arg_r);

            (l_indices, r_indices)
        };

        if l_indices.len() < r_indices.len() {
            ([(arg_r, r_indices), (self.arg_radial, l_indices)], polar_distance)
        } else {
            ([(self.arg_radial, l_indices), (arg_r, r_indices)], polar_distance)
        }
    }

    /// Drops the distances from a vector, returning only the indices.
    fn drop_distances(indices: Vec<((usize, U), U)>) -> Vec<usize> {
        indices.into_iter().map(|((i, _), _)| i).collect()
    }

    /// The `history` of the child `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `right`: Whether the child `Cluster` is the right child.
    fn child_history(&self, right: bool) -> Vec<bool> {
        let mut history = self.history.clone();
        history.push(right);
        history
    }

    /// Sets the chile-parent `Cluster` ratios for anomaly detection and related
    /// applications.
    ///
    /// # Arguments
    ///
    /// * `parent_ratios`: The ratios for the parent `Cluster`.
    #[must_use]
    pub(crate) fn set_child_parent_ratios(mut self, parent_ratios: Ratios) -> Self {
        let [parent_cardinality, parent_radius, parent_lfd, parent_cardinality_ema, parent_radius_ema, parent_lfd_ema] =
            parent_ratios;

        let c = self.cardinality.as_f64() / parent_cardinality;
        let r = self.radius.as_f64() / parent_radius;
        let l = self.lfd / parent_lfd;

        let c_ = utils::next_ema(c, parent_cardinality_ema);
        let r_ = utils::next_ema(r, parent_radius_ema);
        let l_ = utils::next_ema(l, parent_lfd_ema);

        let ratios = [c, r, l, c_, r_, l_];
        self.ratios = Some(ratios);

        if let Some(Children {
            left,
            right,
            arg_l,
            arg_r,
            polar_distance,
        }) = self.children
        {
            let left = Box::new(left.set_child_parent_ratios(ratios));
            let right = Box::new(right.set_child_parent_ratios(ratios));
            let children = Children {
                left,
                right,
                arg_l,
                arg_r,
                polar_distance,
            };
            self.children = Some(children);
        }

        self
    }

    /// Normalizes the `Cluster` ratios for anomaly detection and related
    /// applications.
    ///
    /// # Arguments
    ///
    /// * `means`: The means of the `Cluster` ratios.
    /// * `sds`: The standard deviations of the `Cluster` ratios.
    pub(crate) fn set_normalized_ratios(&mut self, means: Ratios, sds: Ratios) {
        let normalized_ratios: Vec<_> = self
            .ratios
            .unwrap_or_else(|| unreachable!("Ratios should have been set first."))
            .into_iter()
            .zip(means)
            .zip(sds)
            .map(|((value, mean), std)| (value - mean) / std.mul_add(core::f64::consts::SQRT_2, f64::EPSILON))
            .map(libm::erf)
            .map(|v| (1. + v) / 2.)
            .collect();

        if let Ok(normalized_ratios) = normalized_ratios.try_into() {
            self.ratios = Some(normalized_ratios);
        }

        match &mut self.children {
            Some(children) => {
                children.left.set_normalized_ratios(means, sds);
                children.right.set_normalized_ratios(means, sds);
            }
            None => (),
        }
    }

    /// The indices of the `Cluster`'s instances in the dataset.
    pub const fn indices(&self) -> Range<usize> {
        self.offset..(self.offset + self.cardinality)
    }

    /// The `history` of the `Cluster` as a bool vector.
    ///
    /// The root `Cluster` has a `history` of length 1, being `vec![true]`. The
    /// history of a left child is the history of its parent with a `false`
    /// appended. The history of a right child is the history of its parent with
    /// a `true` appended.
    ///
    /// The `history` of a `Cluster` is used to identify the `Cluster` in the
    /// tree, and to compute the `Cluster`'s `name`.
    pub fn history(&self) -> &[bool] {
        &self.history
    }
    /// The `name` of the `Cluster` as a hex-String.
    ///
    /// This is a human-readable representation of the `Cluster`'s `history`.
    /// It may be used to store the `Cluster` in a database, or to identify the
    /// `Cluster` in a visualization.
    pub fn name(&self) -> String {
        Self::history_to_name(&self.history)
    }

    /// Returns a human-readable hexadecimal representation of a `Cluster` history
    /// boolean vector
    #[must_use]
    pub fn history_to_name(history: &[bool]) -> String {
        let rem = history.len() % 4;
        let padding = if rem == 0 { 0 } else { 4 - rem };
        (0..padding) // Pad the history with 0s to make it a multiple of 4.
            .map(|_| &false)
            .chain(history)
            // Convert each bool to a binary string.
            .map(|&b| if b { "1" } else { "0" })
            .collect::<Vec<_>>()
            .chunks_exact(4)
            .map(|s| {
                // Convert each 4-bit binary string to a hexadecimal character.
                u8::from_str_radix(&s.join(""), 2)
                    .unwrap_or_else(|_| unreachable!("We know the characters used are only \"0\" and \"1\"."))
            })
            .fold(String::new(), |mut acc, s| {
                // Append each hexadecimal character to the accumulator.
                write!(&mut acc, "{s:01x}")
                    .unwrap_or_else(|_| unreachable!("We know the characters used are hexadecimal."));
                acc
            })
    }

    /// Returns a boolean vector representation of a `Cluster` history from a
    /// human-readable hexadecimal representation.
    #[must_use]
    pub fn name_to_history(name: &str) -> Vec<bool> {
        name.chars()
            // Convert each hexadecimal character to u8
            .map(|c| {
                u8::from_str_radix(&c.to_string(), 16)
                    .unwrap_or_else(|_| unreachable!("We know the characters used are hexadecimal."))
            })
            .fold(String::new(), |mut acc, c| {
                // Convert each u8 to a 4-bit binary string and append it to the accumulator.
                write!(&mut acc, "{c:04b}")
                    .unwrap_or_else(|_| unreachable!("We know the characters used are only \"0\" and \"1\"."));
                acc
            })
            // Remove any leading 0s.
            .trim_start_matches('0')
            .chars()
            // Convert each binary character to a bool.
            .map(|c| c == '1')
            .collect()
    }

    /// The depth of the `Cluster` in the tree.
    ///
    /// The root `Cluster` has a depth of 0. The depth of a child is the depth
    /// of its parent plus 1.
    pub fn depth(&self) -> usize {
        self.history.len() - 1
    }

    /// Whether the `Cluster` contains only one instance or only identical
    /// instances.
    pub fn is_singleton(&self) -> bool {
        // TODO: How do we handle distance functions that do not obey the
        // identity requirement.
        self.radius == U::zero()
    }

    /// Whether this cluster has no children.
    pub const fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// A 2-slice of references to the left and right child `Cluster`s.
    pub fn children(&self) -> Option<[&Self; 2]> {
        self.children.as_ref().map(|v| [v.left.as_ref(), v.right.as_ref()])
    }

    /// The distance between the poles of the `Cluster`.
    pub fn polar_distance(&self) -> Option<U> {
        self.children.as_ref().map(|v| v.polar_distance)
    }

    /// Whether this `Cluster` is an ancestor of the `other` `Cluster`.
    pub fn is_ancestor_of(&self, other: &Self) -> bool {
        self.depth() < other.depth() && self.history.as_slice() == &other.history[..self.history.len()]
    }

    /// Whether this `Cluster` is an descendant of the `other` `Cluster`.
    pub fn is_descendant_of(&self, other: &Self) -> bool {
        other.is_ancestor_of(self)
    }

    /// A Vec of references to all `Cluster`s in the subtree of this `Cluster`,
    /// including this `Cluster`.
    pub fn subtree(&self) -> Vec<&Self> {
        let subtree = vec![self];

        // Two scenarios: Either we have children or not
        match self.children() {
            Some([left, right]) => subtree
                .into_iter()
                .chain(left.subtree())
                .chain(right.subtree())
                .collect(),

            None => subtree,
        }
    }

    /// The maximum depth of any leaf in the subtree of this `Cluster`.
    pub fn max_leaf_depth(&self) -> usize {
        self.subtree().into_iter().map(Self::depth).max().map_or_else(
            || unreachable!("The subtree of a Cluster should have at least one element, i.e. the Cluster itself."),
            |depth| depth,
        )
    }

    /// Distance from the `center` to the given instance.
    pub fn distance_to_instance<I: Instance, D: Dataset<I, U>>(&self, data: &D, instance: &I) -> U {
        data.query_to_one(instance, self.arg_center)
    }

    /// Distance from the `center` of this `Cluster` to the center of the
    /// `other` `Cluster`.
    pub fn distance_to_other<I: Instance, D: Dataset<I, U>>(&self, data: &D, other: &Self) -> U {
        data.one_to_one(self.arg_center, other.arg_center)
    }

    /// Assuming that this `Cluster` overlaps with with query ball, we return
    /// only those children that also overlap with the query ball
    pub fn overlapping_children<I: Instance, D: Dataset<I, U>>(&self, data: &D, query: &I, radius: U) -> Vec<&Self> {
        self.children.as_ref().map_or_else(
            Vec::new,
            |Children {
                 left,
                 right,
                 arg_l,
                 arg_r,
                 polar_distance,
                 ..
             }| {
                let ql = data.query_to_one(query, *arg_l);
                let qr = data.query_to_one(query, *arg_r);

                let swap = ql < qr;
                let (ql, qr) = if swap { (qr, ql) } else { (ql, qr) };

                if (ql + qr) * (ql - qr) <= U::from(2) * (*polar_distance) * radius {
                    vec![left.as_ref(), right.as_ref()]
                } else if swap {
                    vec![left.as_ref()]
                } else {
                    vec![right.as_ref()]
                }
            },
        )
    }
}

/// Intermediate representation of `Cluster` for serialization.
/// We do this instead of directly serializing/deserializing the Clusters themselves because
/// writing a deserializer directly is moderately complicated with our exceptions
#[allow(clippy::module_name_repetitions)]
#[derive(Serialize, Deserialize)]
pub struct SerializedCluster {
    /// The encoded history of the `Cluster`
    pub name: String,
    /// The seed (if applicable) that the `Cluster` was constructed with
    pub seed: Option<u64>,
    /// The `Cluster`'s offset
    pub offset: usize,
    /// The `Cluster`'s cardinality
    pub cardinality: usize,
    /// The `Cluster`'s arg_center
    pub arg_center: usize,
    /// The `Cluster`'s arg_radial
    pub arg_radial: usize,
    /// The `Cluster`'s radius in byte form
    pub radius_bytes: Vec<u8>,
    /// The `Cluster`'s local fractal dimension
    pub lfd: f64,
    /// The `Cluster`'s ratios
    pub ratios: Option<Ratios>,
}

/// Serialized information about a given `Cluster`'s children
#[derive(Serialize, Deserialize)]
pub struct SerializedChildren {
    /// The left pole of the `Cluster`
    pub arg_l: usize,
    /// The right pole of the `Cluster`
    pub arg_r: usize,
    /// The distance from the `l_pole` to the `r_pole` in bytes.
    /// This value gets reconstituted can be reconstituted from
    /// whatever `U: Number` it was decomposed from
    pub polar_distance_bytes: Vec<u8>,
}

impl SerializedCluster {
    /// Converts a `Cluster` to a `SerializedCluster`
    pub fn from_cluster<U: Number>(cluster: &Cluster<U>) -> (Self, Option<SerializedChildren>) {
        let name = cluster.name();
        let cardinality = cluster.cardinality;
        let offset = cluster.offset;
        let seed = cluster.seed;

        // Because Number isn't serializeable, we just convert it to bytes and
        // serialize that
        let radius_bytes = cluster.radius.to_le_bytes();
        let arg_center = cluster.arg_center;
        let arg_radial = cluster.arg_radial;
        let lfd = cluster.lfd;

        let ratios = cluster.ratios;

        // Since we cant do this recursively we need to like depth-first traverse
        // the tree and serialize manually
        let children = cluster.children.as_ref().map(
            |Children {
                 arg_l,
                 arg_r,
                 polar_distance,
                 ..
             }| {
                SerializedChildren {
                    arg_l: *arg_l,
                    arg_r: *arg_r,
                    polar_distance_bytes: polar_distance.to_le_bytes(),
                }
            },
        );

        (
            Self {
                name,
                seed,
                offset,
                cardinality,
                arg_center,
                arg_radial,
                radius_bytes,
                lfd,
                ratios,
            },
            children,
        )
    }

    /// Converts a `SerializedCluster` to a `Cluster`. Optionally returns information about the
    /// Children's poles
    #[must_use]
    pub fn into_partial_cluster<U: Number>(self) -> Cluster<U> {
        Cluster {
            history: Cluster::<U>::name_to_history(&self.name),
            seed: self.seed,
            offset: self.offset,
            cardinality: self.cardinality,
            arg_center: self.arg_center,
            arg_radial: self.arg_radial,
            radius: U::from_le_bytes(&self.radius_bytes),
            lfd: self.lfd,
            ratios: self.ratios,
            children: None,
        }
    }
}
