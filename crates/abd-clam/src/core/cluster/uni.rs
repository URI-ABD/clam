//! The base struct for all `Cluster` types.

use core::{
    cmp::Ordering,
    fmt::{Display, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
};
use std::time::Instant;

use distances::Number;
use mt_logger::{mt_log, Level};
use serde::{
    de::{MapAccess, SeqAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{utils, Cluster, Dataset, Instance, PartitionCriterion};

use super::Children;

/// A `UniBall` is a cluster that behaves as clusters used to before the introduction
/// of the `Cluster` trait.
///
/// A `UniBall` has a center and a radius, and (optionally) has two children.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct UniBall<U: Number> {
    /// The depth of the `UniBall` in the tree.
    depth: usize,
    /// The offset of the indices of the `UniBall`'s instances in the dataset.
    offset: usize,
    /// The number of instances in the `UniBall`.
    cardinality: usize,
    /// The index of the instance at the `center` of the `UniBall`.
    arg_center: usize,
    /// The index of the instance with the maximum distance from the `center`
    arg_radial: usize,
    /// The radius of the `UniBall`.
    radius: U,
    /// The local fractal dimension of the `UniBall`.
    lfd: f64,
    /// The children of the `UniBall`.
    pub(crate) children: Option<Children<U, Self>>,
}

impl<U: Number> PartialEq for UniBall<U> {
    fn eq(&self, other: &Self) -> bool {
        self.offset == other.offset && self.cardinality == other.cardinality
    }
}

impl<U: Number> Eq for UniBall<U> {}

impl<U: Number> PartialOrd for UniBall<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: Number> Ord for UniBall<U> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.depth().cmp(&other.depth()) {
            Ordering::Equal => self.offset.cmp(&other.offset),
            ordering => ordering,
        }
    }
}

impl<U: Number> Hash for UniBall<U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.offset, self.cardinality).hash(state);
    }
}

impl<U: Number> Display for UniBall<U> {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl<U: Number> UniBall<U> {
    /// Create a new `UniBall`.
    fn new<I: Instance, D: Dataset<I, U>>(
        data: &D,
        seed: Option<u64>,
        offset: usize,
        indices: &[usize],
        depth: usize,
    ) -> Self {
        let cardinality = indices.len();

        let start = Instant::now();
        mt_log!(
            Level::Debug,
            "Creating a UniBall with depth {depth} and cardinality {cardinality} ..."
        );

        let arg_samples = if cardinality < 100 {
            indices.to_vec()
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let n = (cardinality.as_f64().sqrt()) as usize;
            data.choose_unique(n, indices, seed)
        };

        let Some(arg_center) = data.median(&arg_samples) else {
            unreachable!("The UniBall has at least one instance.")
        };

        let center_distances = data.one_to_many(arg_center, indices);
        let Some((arg_radial, radius)) = utils::arg_max(&center_distances).map(|(i, r)| (indices[i], r)) else {
            unreachable!("The UniBall has at least one instance.")
        };

        let lfd = utils::compute_lfd(radius, &center_distances);

        let end = start.elapsed().as_secs_f32();
        mt_log!(
            Level::Debug,
            "Finished creating a UniBall with depth {depth}, offset {offset} and cardinality {cardinality} in {end:.2e} seconds."
        );

        Self {
            depth,
            offset,
            cardinality,
            arg_center,
            arg_radial,
            radius,
            lfd,
            children: None,
        }
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
    ///      cardinality of the `UniBall`.
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
    fn _partition<I: Instance, D: Dataset<I, U>, P: PartitionCriterion<U>>(
        mut self,
        data: &D,
        criteria: &P,
        mut indices: Vec<usize>,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>) {
        if criteria.check(&self) {
            let ([(arg_l, l_indices), (arg_r, r_indices)], polar_distance) = self.partition_once(data, indices.clone());
            if self._check_partition(&l_indices, &r_indices) {
                core::mem::drop(indices);

                let r_offset = self.offset + l_indices.len();

                let ((left, l_indices), (right, r_indices)) = rayon::join(
                    || {
                        Self::new(data, seed, self.offset, &l_indices, self.depth + 1)
                            ._partition(data, criteria, l_indices, seed)
                    },
                    || {
                        Self::new(data, seed, r_offset, &r_indices, self.depth + 1)
                            ._partition(data, criteria, r_indices, seed)
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

    /// Partitions the `UniBall` into two children once.
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
}

impl<U: Number> Cluster<U> for UniBall<U> {
    fn new_root<I: Instance, D: Dataset<I, U>>(data: &D, seed: Option<u64>) -> Self {
        let indices = (0..data.cardinality()).collect::<Vec<usize>>();
        Self::new(data, seed, 0, &indices, 0)
    }

    fn partition<I: Instance, D: Dataset<I, U>, P: PartitionCriterion<U>>(
        mut self,
        data: &mut D,
        criteria: &P,
        seed: Option<u64>,
    ) -> Self {
        let mut indices = (0..self.cardinality).collect::<Vec<_>>();
        (self, indices) = self._partition(data, criteria, indices, seed);

        mt_log!(Level::Debug, "Finished building tree. Starting data permutation.");
        data.permute_instances(&indices).unwrap_or_else(|e| unreachable!("{e}"));
        mt_log!(Level::Debug, "Finished data permutation.");

        self
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn cardinality(&self) -> usize {
        self.cardinality
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn arg_center(&self) -> usize {
        self.arg_center
    }

    fn radius(&self) -> U {
        self.radius
    }

    fn arg_radial(&self) -> usize {
        self.arg_radial
    }

    fn lfd(&self) -> f64 {
        self.lfd
    }

    fn children(&self) -> Option<[&Self; 2]> {
        self.children.as_ref().map(|c| [c.left.as_ref(), c.right.as_ref()])
    }

    fn polar_distance(&self) -> Option<U> {
        self.children.as_ref().map(|c| c.polar_distance)
    }

    fn arg_poles(&self) -> Option<[usize; 2]> {
        self.children.as_ref().map(|c| [c.arg_l, c.arg_r])
    }
}

impl<U: Number> Serialize for UniBall<U> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("UniBall", 8)?;
        state.serialize_field("depth", &self.depth)?;
        state.serialize_field("offset", &self.offset)?;
        state.serialize_field("cardinality", &self.cardinality)?;
        state.serialize_field("arg_center", &self.arg_center)?;
        state.serialize_field("arg_radial", &self.arg_radial)?;
        state.serialize_field("radius", &self.radius.to_le_bytes())?;
        state.serialize_field("lfd", &self.lfd)?;
        state.serialize_field("children", &self.children)?;
        state.end()
    }
}

impl<'de, U: Number> Deserialize<'de> for UniBall<U> {
    #[allow(clippy::too_many_lines)]
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// The fields in the `UniBall` struct.
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            /// The depth of this `UniBall` in the tree.
            Depth,
            /// The offset of the indices of the `UniBall`'s instances in the dataset.
            Offset,
            /// The number of instances in the `UniBall`.
            Cardinality,
            /// The index of the `center` instance in the dataset.
            ArgCenter,
            /// The index of the `radial` instance in the dataset.
            ArgRadial,
            /// The distance from the `center` to the `radial` instance.
            Radius,
            /// The local fractal dimension of the `UniBall`.
            Lfd,
            /// The children of the `UniBall`.
            Children,
        }

        /// The `UniBall` visitor for deserialization.
        struct UniBallVisitor<U: Number>(PhantomData<U>);

        impl<'de, U: Number> Visitor<'de> for UniBallVisitor<U> {
            type Value = UniBall<U>;

            fn expecting(&self, formatter: &mut Formatter) -> core::fmt::Result {
                formatter.write_str("struct UniBall")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let depth = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
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

                Ok(UniBall {
                    depth,
                    offset,
                    cardinality,
                    arg_center,
                    arg_radial,
                    radius,
                    lfd,
                    children,
                })
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Self::Value, V::Error> {
                let mut depth = None;
                let mut offset = None;
                let mut cardinality = None;
                let mut arg_center = None;
                let mut arg_radial = None;
                let mut radius = None;
                let mut lfd = None;
                let mut children = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Depth => {
                            if depth.is_some() {
                                return Err(serde::de::Error::duplicate_field("depth"));
                            }
                            depth = Some(map.next_value()?);
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
                    }
                }

                let depth = depth.ok_or_else(|| serde::de::Error::missing_field("depth"))?;
                let offset = offset.ok_or_else(|| serde::de::Error::missing_field("offset"))?;
                let cardinality = cardinality.ok_or_else(|| serde::de::Error::missing_field("cardinality"))?;
                let arg_center = arg_center.ok_or_else(|| serde::de::Error::missing_field("arg_center"))?;
                let arg_radial = arg_radial.ok_or_else(|| serde::de::Error::missing_field("arg_radial"))?;

                let radius_bytes: Vec<u8> = radius.ok_or_else(|| serde::de::Error::missing_field("radius"))?;
                let radius = U::from_le_bytes(&radius_bytes);

                let lfd = lfd.ok_or_else(|| serde::de::Error::missing_field("lfd"))?;
                let children = children.ok_or_else(|| serde::de::Error::missing_field("children"))?;

                Ok(UniBall {
                    depth,
                    offset,
                    cardinality,
                    arg_center,
                    arg_radial,
                    radius,
                    lfd,
                    children,
                })
            }
        }

        /// The fields in the `UniBall` struct.
        const FIELDS: &[&str] = &[
            "depth",
            "offset",
            "cardinality",
            "arg_center",
            "arg_radial",
            "radius",
            "lfd",
            "children",
        ];
        deserializer.deserialize_struct("UniBall", FIELDS, UniBallVisitor(PhantomData))
    }
}
