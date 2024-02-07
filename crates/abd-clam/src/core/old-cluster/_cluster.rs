//! The `Cluster` is the heart of CLAM. It provides the ability to perform a
//! divisive hierarchical cluster of arbitrary datasets in arbitrary metric
//! spaces.

use core::{
    cmp::Ordering,
    fmt::{Display, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::Range,
};

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use distances::Number;
use mt_logger::{mt_log, Level};
use serde::{
    de::{MapAccess, SeqAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

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
    /// The depth of this `Cluster` in the tree.
    depth: usize,
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
    ratios: Option<Ratios>,
}

impl<U: Number> Serialize for Cluster<U> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Cluster", 10)?;
        state.serialize_field("depth", &self.depth)?;
        state.serialize_field("seed", &self.seed)?;
        state.serialize_field("offset", &self.offset)?;
        state.serialize_field("cardinality", &self.cardinality)?;
        state.serialize_field("arg_center", &self.arg_center)?;
        state.serialize_field("arg_radial", &self.arg_radial)?;
        state.serialize_field("radius", &self.radius.to_le_bytes())?;
        state.serialize_field("lfd", &self.lfd)?;
        state.serialize_field("children", &self.children)?;
        state.serialize_field("ratios", &self.ratios)?;
        state.end()
    }
}

impl<'de, U: Number> Deserialize<'de> for Cluster<U> {
    #[allow(clippy::too_many_lines)]
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// The fields in the `Cluster` struct.
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            /// The depth of this `Cluster` in the tree.
            Depth,
            /// The seed used in the random number generator for this `Cluster`.
            Seed,
            /// The offset of the indices of the `Cluster`'s instances in the dataset.
            Offset,
            /// The number of instances in the `Cluster`.
            Cardinality,
            /// The index of the `center` instance in the dataset.
            ArgCenter,
            /// The index of the `radial` instance in the dataset.
            ArgRadial,
            /// The distance from the `center` to the `radial` instance.
            Radius,
            /// The local fractal dimension of the `Cluster`.
            Lfd,
            /// The children of the `Cluster`.
            Children,
            /// The six `Cluster` ratios used for anomaly detection and related applications.
            Ratios,
        }

        /// The `Cluster` visitor for deserialization.
        struct ClusterVisitor<U: Number>(PhantomData<U>);

        impl<'de, U: Number> Visitor<'de> for ClusterVisitor<U> {
            type Value = Cluster<U>;

            fn expecting(&self, formatter: &mut Formatter) -> core::fmt::Result {
                formatter.write_str("struct Cluster")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let depth = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let seed = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let offset = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                let cardinality = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;
                let arg_center = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(4, &self))?;
                let arg_radial = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(5, &self))?;

                let radius_bytes: Vec<u8> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(6, &self))?;
                let radius = U::from_le_bytes(&radius_bytes);

                let lfd = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(7, &self))?;
                let children = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(8, &self))?;
                let ratios = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(9, &self))?;

                Ok(Cluster {
                    depth,
                    seed,
                    offset,
                    cardinality,
                    arg_center,
                    arg_radial,
                    radius,
                    lfd,
                    children,
                    ratios,
                })
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Self::Value, V::Error> {
                let mut depth = None;
                let mut seed = None;
                let mut offset = None;
                let mut cardinality = None;
                let mut arg_center = None;
                let mut arg_radial = None;
                let mut radius = None;
                let mut lfd = None;
                let mut children = None;
                let mut ratios = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Depth => {
                            if depth.is_some() {
                                return Err(serde::de::Error::duplicate_field("depth"));
                            }
                            depth = Some(map.next_value()?);
                        }
                        Field::Seed => {
                            if seed.is_some() {
                                return Err(serde::de::Error::duplicate_field("seed"));
                            }
                            seed = Some(map.next_value()?);
                        }
                        Field::Offset => {
                            if offset.is_some() {
                                return Err(serde::de::Error::duplicate_field("offset"));
                            }
                            offset = Some(map.next_value()?);
                        }
                        Field::Cardinality => {
                            if cardinality.is_some() {
                                return Err(serde::de::Error::duplicate_field("cardinality"));
                            }
                            cardinality = Some(map.next_value()?);
                        }
                        Field::ArgCenter => {
                            if arg_center.is_some() {
                                return Err(serde::de::Error::duplicate_field("arg_center"));
                            }
                            arg_center = Some(map.next_value()?);
                        }
                        Field::ArgRadial => {
                            if arg_radial.is_some() {
                                return Err(serde::de::Error::duplicate_field("arg_radial"));
                            }
                            arg_radial = Some(map.next_value()?);
                        }
                        Field::Radius => {
                            if radius.is_some() {
                                return Err(serde::de::Error::duplicate_field("radius"));
                            }
                            radius = Some(map.next_value()?);
                        }
                        Field::Lfd => {
                            if lfd.is_some() {
                                return Err(serde::de::Error::duplicate_field("lfd"));
                            }
                            lfd = Some(map.next_value()?);
                        }
                        Field::Children => {
                            if children.is_some() {
                                return Err(serde::de::Error::duplicate_field("children"));
                            }
                            children = Some(map.next_value()?);
                        }
                        Field::Ratios => {
                            if ratios.is_some() {
                                return Err(serde::de::Error::duplicate_field("ratios"));
                            }
                            ratios = Some(map.next_value()?);
                        }
                    }
                }

                let depth = depth.ok_or_else(|| serde::de::Error::missing_field("depth"))?;
                let seed = seed.ok_or_else(|| serde::de::Error::missing_field("seed"))?;
                let offset = offset.ok_or_else(|| serde::de::Error::missing_field("offset"))?;
                let cardinality = cardinality.ok_or_else(|| serde::de::Error::missing_field("cardinality"))?;
                let arg_center = arg_center.ok_or_else(|| serde::de::Error::missing_field("arg_center"))?;
                let arg_radial = arg_radial.ok_or_else(|| serde::de::Error::missing_field("arg_radial"))?;

                let radius_bytes: Vec<u8> = radius.ok_or_else(|| serde::de::Error::missing_field("radius"))?;
                let radius = U::from_le_bytes(&radius_bytes);

                let lfd = lfd.ok_or_else(|| serde::de::Error::missing_field("lfd"))?;
                let children = children.ok_or_else(|| serde::de::Error::missing_field("children"))?;
                let ratios = ratios.ok_or_else(|| serde::de::Error::missing_field("ratios"))?;

                Ok(Cluster {
                    depth,
                    seed,
                    offset,
                    cardinality,
                    arg_center,
                    arg_radial,
                    radius,
                    lfd,
                    children,
                    ratios,
                })
            }
        }

        /// The fields in the `Cluster` struct.
        const FIELDS: &[&str] = &[
            "depth",
            "seed",
            "offset",
            "cardinality",
            "arg_center",
            "arg_radial",
            "radius",
            "lfd",
            "children",
            "ratios",
        ];
        deserializer.deserialize_struct("Cluster", FIELDS, ClusterVisitor(PhantomData))
    }
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

impl<U: Number> Serialize for Children<U> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Children", 5)?;
        state.serialize_field("left", &self.left)?;
        state.serialize_field("right", &self.right)?;
        state.serialize_field("arg_l", &self.arg_l)?;
        state.serialize_field("arg_r", &self.arg_r)?;
        state.serialize_field("polar_distance", &self.polar_distance.to_le_bytes())?;
        state.end()
    }
}

impl<'de, U: Number> Deserialize<'de> for Children<U> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// The fields in the `Children` struct.
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            /// The left child of the `Cluster`.
            Left,
            /// The right child of the `Cluster`.
            Right,
            /// The left pole of the `Cluster` (i.e. the instance used to identify
            ArgL,
            /// The right pole of the `Cluster` (i.e. the instance used to identify
            ArgR,
            /// The distance from the `l_pole` to the `r_pole` instance.
            PolarDistance,
        }

        /// The `Children` visitor for deserialization.
        struct ChildrenVisitor<U: Number>(PhantomData<U>);

        impl<'de, U: Number> Visitor<'de> for ChildrenVisitor<U> {
            type Value = Children<U>;

            fn expecting(&self, formatter: &mut Formatter) -> core::fmt::Result {
                formatter.write_str("struct Children")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let left = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let right = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let arg_l = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                let arg_r = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;

                let polar_distance_bytes: Vec<u8> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(4, &self))?;
                let polar_distance = U::from_le_bytes(&polar_distance_bytes);

                Ok(Children {
                    left,
                    right,
                    arg_l,
                    arg_r,
                    polar_distance,
                })
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Self::Value, V::Error> {
                let mut left = None;
                let mut right = None;
                let mut arg_l = None;
                let mut arg_r = None;
                let mut polar_distance = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Left => {
                            if left.is_some() {
                                return Err(serde::de::Error::duplicate_field("left"));
                            }
                            left = Some(map.next_value()?);
                        }
                        Field::Right => {
                            if right.is_some() {
                                return Err(serde::de::Error::duplicate_field("right"));
                            }
                            right = Some(map.next_value()?);
                        }
                        Field::ArgL => {
                            if arg_l.is_some() {
                                return Err(serde::de::Error::duplicate_field("arg_l"));
                            }
                            arg_l = Some(map.next_value()?);
                        }
                        Field::ArgR => {
                            if arg_r.is_some() {
                                return Err(serde::de::Error::duplicate_field("arg_r"));
                            }
                            arg_r = Some(map.next_value()?);
                        }
                        Field::PolarDistance => {
                            if polar_distance.is_some() {
                                return Err(serde::de::Error::duplicate_field("polar_distance"));
                            }
                            polar_distance = Some(map.next_value()?);
                        }
                    }
                }

                let left = left.ok_or_else(|| serde::de::Error::missing_field("left"))?;
                let right = right.ok_or_else(|| serde::de::Error::missing_field("right"))?;
                let arg_l = arg_l.ok_or_else(|| serde::de::Error::missing_field("arg_l"))?;
                let arg_r = arg_r.ok_or_else(|| serde::de::Error::missing_field("arg_r"))?;

                let polar_distance_bytes: Vec<u8> =
                    polar_distance.ok_or_else(|| serde::de::Error::missing_field("polar_distance"))?;
                let polar_distance = U::from_le_bytes(&polar_distance_bytes);

                Ok(Children {
                    left,
                    right,
                    arg_l,
                    arg_r,
                    polar_distance,
                })
            }
        }

        /// The fields in the `Children` struct.
        const FIELDS: &[&str] = &["left", "right", "arg_l", "arg_r", "polar_distance"];
        deserializer.deserialize_struct("Children", FIELDS, ChildrenVisitor(PhantomData))
    }
}

impl<U: Number> PartialEq for Cluster<U> {
    fn eq(&self, other: &Self) -> bool {
        self.offset == other.offset && self.cardinality == other.cardinality
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
    /// The offset of the indices of the `Cluster`'s instances in the dataset.
    pub const fn offset(&self) -> usize {
        self.offset
    }

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
        Self::new(data, seed, 0, &indices, 0)
    }

    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `data`: on which to create the `Cluster`.
    /// * `seed`: The seed used in the random number generator for this `Cluster`.
    /// * `offset`: The offset of the indices of the `Cluster`'s instances in the dataset.
    /// * `indices`: The indices of instances from the `dataset` that are contained in the `Cluster`.
    /// * `depth`: The depth of the `Cluster` in the tree.
    fn new<I: Instance, D: Dataset<I, U>>(
        data: &D,
        seed: Option<u64>,
        offset: usize,
        indices: &[usize],
        depth: usize,
    ) -> Self {
        let cardinality = indices.len();

        let start = std::time::Instant::now();
        mt_log!(
            Level::Debug,
            "Creating cluster with depth {depth} offset {offset} and cardinality {cardinality} ..."
        );

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

        let end = start.elapsed().as_secs_f32();
        mt_log!(
            Level::Debug,
            "Finished creating cluster with depth {depth} offset {offset} and cardinality {cardinality} in {end:2e} seconds."
        );

        Self {
            depth,
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

        mt_log!(Level::Debug, "Finished building tree. Starting data permutation.");
        data.permute_instances(&indices).unwrap_or_else(|e| unreachable!("{e}"));
        mt_log!(Level::Debug, "Finished data permutation.");

        self
    }

    /// Checks that the partition is valid.
    ///
    /// # Arguments
    ///
    /// * `l_indices`: The indices of the left child.
    /// * `r_indices`: The indices of the right child.
    ///
    /// # Returns
    ///
    /// * `true` if all of the following conditions are met:
    ///    * The `l_indices` are not empty.
    ///    * The `r_indices` are not empty.
    ///    * The total length of the `l_indices` and `r_indices` is equal to the
    ///      cardinality of the `Cluster`.
    const fn _check_partition(&self, l_indices: &[usize], r_indices: &[usize]) -> bool {
        !l_indices.is_empty() && !r_indices.is_empty() && l_indices.len() + r_indices.len() == self.cardinality
        // assert!(
        //     !l_indices.is_empty(),
        //     "Left child of {} at depth {} should not be empty.",
        //     self.name(),
        //     self.depth
        // );
        // assert!(
        //     !r_indices.is_empty(),
        //     "Right child of {} at depth {} should not be empty.",
        //     self.name(),
        //     self.depth
        // );
        // assert!(
        //     l_indices.len() + r_indices.len() == self.cardinality,
        //     "The sum of the left and right child of {} at depth {} should be equal to the cardinality of the cluster.",
        //     self.name(),
        //     self.depth
        // );
    }

    /// Recursive helper function for `partition`.
    fn _partition<I: Instance, D: Dataset<I, U>>(
        mut self,
        data: &D,
        criteria: &PartitionCriteria<U>,
        mut indices: Vec<usize>,
    ) -> (Self, Vec<usize>) {
        if criteria.check(&self) {
            let ([(arg_l, l_indices), (arg_r, r_indices)], polar_distance) = self.partition_once(data, indices.clone());
            if self._check_partition(&l_indices, &r_indices) {
                core::mem::drop(indices);

                let r_offset = self.offset + l_indices.len();

                let ((left, l_indices), (right, r_indices)) = rayon::join(
                    || {
                        Self::new(data, self.seed, self.offset, &l_indices, self.depth + 1)
                            ._partition(data, criteria, l_indices)
                    },
                    || {
                        Self::new(data, self.seed, r_offset, &r_indices, self.depth + 1)
                            ._partition(data, criteria, r_indices)
                    },
                );
                self._check_partition(&l_indices, &r_indices);

                let arg_l = utils::position_of(&l_indices, arg_l)
                    .unwrap_or_else(|| unreachable!("We know the left pole is in the indices."));
                let arg_r = utils::position_of(&r_indices, arg_r)
                    .unwrap_or_else(|| unreachable!("We know the right pole is in the indices."));

                self.children = Some(Children {
                    left: Box::new(left),
                    right: Box::new(right),
                    arg_l: self.offset + arg_l,
                    arg_r: r_offset + arg_r,
                    polar_distance,
                });

                indices = l_indices.into_iter().chain(r_indices).collect::<Vec<_>>();
            }
        }

        // reset the indices to center and radial indices for data reordering
        let arg_center = utils::position_of(&indices, self.arg_center)
            .unwrap_or_else(|| unreachable!("We know the center is in the indices."));
        self.arg_center = self.offset + arg_center;

        let arg_radial = utils::position_of(&indices, self.arg_radial)
            .unwrap_or_else(|| unreachable!("We know the radial is in the indices."));
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

    /// Descends to the `Cluster` with the given `offset` and `cardinality`.
    ///
    /// If such a `Cluster` does not exist, `None` is returned.
    ///
    /// # Arguments
    ///
    /// * `offset`: The offset of the `Cluster`'s instances in the dataset.
    /// * `cardinality`: The number of instances in the `Cluster`.
    pub(crate) fn descend_to(&self, offset: usize, cardinality: usize) -> Option<&Self> {
        if self.offset == offset && self.cardinality == cardinality {
            Some(self)
        } else {
            self.children().and_then(|[left, right]| {
                if left.indices().contains(&offset) {
                    left.descend_to(offset, cardinality)
                } else {
                    right.descend_to(offset, cardinality)
                }
            })
        }
    }

    /// The indices of the `Cluster`'s instances in the dataset.
    pub const fn indices(&self) -> Range<usize> {
        self.offset..(self.offset + self.cardinality)
    }

    /// The `name` of the `Cluster` as a hex-String.
    ///
    /// This is a human-readable representation of the `Cluster`'s `offset` and `cardinality`.
    /// It is a unique identifier in the tree.
    /// It may be used to store the `Cluster` in a database, or to identify the
    /// `Cluster` in a visualization.
    pub fn name(&self) -> String {
        format!("{}-{}", self.offset, self.cardinality)
    }

    /// The depth of the `Cluster` in the tree.
    ///
    /// The root `Cluster` has a depth of 0. The depth of a child is the depth
    /// of its parent plus 1.
    pub const fn depth(&self) -> usize {
        self.depth
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
        self.cardinality > other.cardinality && self.indices().contains(&other.offset)
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
    pub fn save(&self, path: &Path) -> Result<(), String> {
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
    pub fn load(path: &Path) -> Result<Self, String> {
        let reader = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
        bincode::deserialize_from(reader).map_err(|e| e.to_string())
    }
}
