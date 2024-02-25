//! Generic `Children` struct for a cluster.

use core::{
    fmt::{Display, Formatter},
    marker::PhantomData,
};

use distances::Number;
use serde::{
    de::{MapAccess, SeqAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize,
};

use crate::Cluster;

/// The `Children` of a `Cluster`.
#[derive(Debug)]
pub struct Children<U: Number, C: Cluster<U>> {
    /// The left child of the `Cluster`.
    pub left: Box<C>,
    /// The right child of the `Cluster`.
    pub right: Box<C>,
    /// The left pole of the `Cluster` (i.e. the instance used to identify
    /// instances for the left child).
    pub arg_l: usize,
    /// The right pole of the `Cluster` (i.e. the instance used to identify
    /// instances for the right child).
    pub arg_r: usize,
    /// The distance from the `l_pole` to the `r_pole` instance.
    pub polar_distance: U,
}

impl<U: Number, C: Cluster<U>> Display for Children<U, C> {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "{} x {}", self.left.name(), self.right.name())
    }
}

impl<U: Number, C: Cluster<U>> Serialize for Children<U, C> {
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

impl<'de, U: Number, C: Cluster<U>> Deserialize<'de> for Children<U, C> {
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
        struct ChildrenVisitor<U: Number, C: Cluster<U>>((PhantomData<U>, PhantomData<C>));

        impl<'de, U: Number, C: Cluster<U>> Visitor<'de> for ChildrenVisitor<U, C> {
            type Value = Children<U, C>;

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
        deserializer.deserialize_struct("Children", FIELDS, ChildrenVisitor((PhantomData, PhantomData)))
    }
}
