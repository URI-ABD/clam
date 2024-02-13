//! A `SquishedBall` is a `Cluster` that supports compression.

use core::{
    cmp::Ordering,
    fmt::{Display, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use distances::Number;
use serde::{
    de::{MapAccess, SeqAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{core::cluster::Children, BaseCluster, Cluster, Dataset, Instance, PartitionCriterion};

/// A `SquishedBall` is a `Cluster` that supports compression.
#[derive(Debug)]
pub struct SquishedBall<U: Number> {
    /// The `BaseCluster` for the underlying `Cluster`.
    base_cluster: BaseCluster<U>,
    /// Expected memory cost, in bytes, of recursive compression.
    recursive_cost: usize,
    /// Expected memory cost, in bytes, of unitary compression.
    unitary_cost: usize,
    /// Child Clusters
    children: Option<Children<U, Self>>,
}

impl<U: Number> SquishedBall<U> {
    /// Creates a new `Vertex`.
    pub const fn new(
        base_cluster: BaseCluster<U>,
        recursive_cost: usize,
        unitary_cost: usize,
        children: Option<Children<U, Self>>,
    ) -> Self {
        Self {
            base_cluster,
            recursive_cost,
            unitary_cost,
            children,
        }
    }

    /// Creates a new `Vertex` tree.
    pub const fn from_base_tree(base_cluster: BaseCluster<U>) -> Self {
        // TODO: Calculate the costs and children
        let recursive_cost = 0;
        let unitary_cost = 0;
        Self::new(base_cluster, recursive_cost, unitary_cost, None)
    }

    /// Normalizes the costs in the subtree.
    #[must_use]
    pub fn normalize_costs(self) -> Self {
        todo!()
    }

    /// The base `BaseCluster` of the `Vertex`.
    pub const fn base_cluster(&self) -> &BaseCluster<U> {
        &self.base_cluster
    }

    /// Returns the expected memory cost, in bytes, of recursive compression.
    pub const fn recursive_cost(&self) -> usize {
        self.recursive_cost
    }

    /// Returns the expected memory cost, in bytes, of unitary compression.
    pub const fn unitary_cost(&self) -> usize {
        self.unitary_cost
    }

    /// Compresses the `SquishedBall` recursively by descending a level into the subtree.
    ///
    /// The centers of the children are encoded in terms of the center of the `SquishedBall`.
    #[must_use]
    pub fn compress_recursive(&self) -> Self {
        // TODO: Implement encode/decode for metrics, probably as an extension trait for `Dataset`.
        todo!()
    }

    /// Compresses the `SquishedBall` without descending into the subtree.
    ///
    /// Every `Instance` is encoded in terms of the center of the `SquishedBall`.
    #[must_use]
    pub fn compress_unitary(&self) -> Self {
        // TODO: Implement encode/decode for metrics, probably as an extension trait for `Dataset`.
        todo!()
    }
}

impl<U: Number> Cluster<U> for SquishedBall<U> {
    fn new_root<I: Instance, D: Dataset<I, U>>(data: &D, seed: Option<u64>) -> Self {
        let base_cluster = BaseCluster::new_root(data, seed);
        Self::new(base_cluster, 0, 0, None)
    }

    #[allow(unused_variables)]
    fn partition<I, D, P>(self, data: &mut D, criteria: &P, seed: Option<u64>) -> Self
    where
        I: Instance,
        D: Dataset<I, U>,
        P: PartitionCriterion<U>,
    {
        let base_cluster = self.base_cluster.partition(data, criteria, seed);
        Self::from_base_tree(base_cluster)
    }

    fn offset(&self) -> usize {
        self.base_cluster.offset()
    }

    fn cardinality(&self) -> usize {
        self.base_cluster.cardinality()
    }

    fn depth(&self) -> usize {
        self.base_cluster.depth()
    }

    fn arg_center(&self) -> usize {
        self.base_cluster.arg_center()
    }

    fn radius(&self) -> U {
        self.base_cluster.radius()
    }

    fn arg_radial(&self) -> usize {
        self.base_cluster.arg_radial()
    }

    fn lfd(&self) -> f64 {
        self.base_cluster.lfd()
    }

    fn children(&self) -> Option<[&Self; 2]> {
        self.children.as_ref().map(|c| [c.left.as_ref(), c.right.as_ref()])
    }

    fn polar_distance(&self) -> Option<U> {
        self.base_cluster.polar_distance()
    }

    fn arg_poles(&self) -> Option<[usize; 2]> {
        self.base_cluster.arg_poles()
    }
}

impl<U: Number> PartialEq for SquishedBall<U> {
    fn eq(&self, other: &Self) -> bool {
        self.base_cluster.eq(&other.base_cluster)
    }
}

impl<U: Number> Eq for SquishedBall<U> {}

impl<U: Number> PartialOrd for SquishedBall<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: Number> Ord for SquishedBall<U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.base_cluster.cmp(&other.base_cluster)
    }
}

impl<U: Number> Hash for SquishedBall<U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base_cluster.hash(state);
    }
}

impl<U: Number> Display for SquishedBall<U> {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "{}", self.base_cluster)
    }
}

impl<U: Number> Serialize for SquishedBall<U> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("SquishedBall", 4)?;
        state.serialize_field("base_cluster", &self.base_cluster)?;
        state.serialize_field("recursive_cost", &self.recursive_cost)?;
        state.serialize_field("unitary_cost", &self.unitary_cost)?;
        state.serialize_field("children", &self.children)?;
        state.end()
    }
}

impl<'de, U: Number> Deserialize<'de> for SquishedBall<U> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// The fields in the `SquishedBall` struct.
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            /// The base `BaseCluster` of the `SquishedBall`.
            BaseCluster,
            /// The expected memory cost, in bytes, of recursive compression.
            RecursiveCost,
            /// The expected memory cost, in bytes, of unitary compression.
            UnitaryCost,
            /// The children of the `SquishedBall`.
            Children,
        }

        /// The `Visitor` for the `SquishedBall` struct.
        struct SquishedBallVisitor<U: Number>(PhantomData<U>);

        impl<'de, U: Number> Visitor<'de> for SquishedBallVisitor<U> {
            type Value = SquishedBall<U>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("struct SquishedBall")
            }

            fn visit_seq<V: SeqAccess<'de>>(self, mut seq: V) -> Result<Self::Value, V::Error> {
                let base_cluster = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let recursive_cost = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let unitary_cost = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                let children = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;
                Ok(SquishedBall::new(base_cluster, recursive_cost, unitary_cost, children))
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Self::Value, V::Error> {
                let mut base_cluster = None;
                let mut recursive_cost = None;
                let mut unitary_cost = None;
                let mut children = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::BaseCluster => {
                            if base_cluster.is_some() {
                                return Err(serde::de::Error::duplicate_field("base_cluster"));
                            }
                            base_cluster = Some(map.next_value()?);
                        }
                        Field::RecursiveCost => {
                            if recursive_cost.is_some() {
                                return Err(serde::de::Error::duplicate_field("recursive_cost"));
                            }
                            recursive_cost = Some(map.next_value()?);
                        }
                        Field::UnitaryCost => {
                            if unitary_cost.is_some() {
                                return Err(serde::de::Error::duplicate_field("unitary_cost"));
                            }
                            unitary_cost = Some(map.next_value()?);
                        }
                        Field::Children => {
                            if children.is_some() {
                                return Err(serde::de::Error::duplicate_field("children"));
                            }
                            children = Some(map.next_value()?);
                        }
                    }
                }

                let base_cluster = base_cluster.ok_or_else(|| serde::de::Error::missing_field("base_cluster"))?;
                let recursive_cost = recursive_cost.ok_or_else(|| serde::de::Error::missing_field("recursive_cost"))?;
                let unitary_cost = unitary_cost.ok_or_else(|| serde::de::Error::missing_field("unitary_cost"))?;
                let children = children.ok_or_else(|| serde::de::Error::missing_field("children"))?;

                Ok(SquishedBall::new(base_cluster, recursive_cost, unitary_cost, children))
            }
        }

        /// The `Field` names.
        const FIELDS: &[&str] = &["base_cluster", "recursive_cost", "unitary_cost", "children"];
        deserializer.deserialize_struct("SquishedBall", FIELDS, SquishedBallVisitor(PhantomData))
    }
}
