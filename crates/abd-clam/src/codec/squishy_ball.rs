//! A `SquishyBall` is a `Cluster` that supports compression.

use core::{
    cmp::Ordering,
    fmt::{Display, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use distances::{number::Int, Number};
use serde::{
    de::{MapAccess, SeqAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{core::cluster::Children, Cluster, Dataset, Instance, PartitionCriterion, UniBall};

use super::SquishyDataset;

/// A `SquishyBall` is a `Cluster` that supports compression.
#[derive(Debug)]
pub struct SquishyBall<U: Int> {
    /// The `UniBall` for the underlying `Cluster`.
    uni_ball: UniBall<U>,
    /// Expected memory cost, in bytes, of recursive compression.
    recursive_cost: u64,
    /// Expected memory cost, in bytes, of unitary compression.
    unitary_cost: u64,
    /// Child Clusters
    children: Option<Children<U, Self>>,
}

impl<U: Int> SquishyBall<U> {
    /// Creates a new `SquishyBall` tree.
    pub fn from_base_tree(root: UniBall<U>) -> Self {
        Self::from_uni_ball(root)
    }

    /// Recursively creates a new `SquishyBall` tree.
    fn from_uni_ball(mut uni_ball: UniBall<U>) -> Self {
        match uni_ball.children {
            Some(children) => {
                uni_ball.children = None;
                let left = Box::new(Self::from_uni_ball(*children.left));
                let right = Box::new(Self::from_uni_ball(*children.right));
                let children = Children {
                    left,
                    right,
                    arg_l: children.arg_l,
                    arg_r: children.arg_r,
                    polar_distance: children.polar_distance,
                };
                Self {
                    uni_ball,
                    recursive_cost: 0,
                    unitary_cost: 0,
                    children: Some(children),
                }
            }
            None => Self {
                uni_ball,
                recursive_cost: 0,
                unitary_cost: 0,
                children: None,
            },
        }
    }

    /// The base `UniBall` of the `Vertex`.
    pub const fn uni_ball(&self) -> &UniBall<U> {
        &self.uni_ball
    }

    /// Returns the expected memory cost, in bytes, of recursive compression.
    pub const fn recursive_cost(&self) -> u64 {
        self.recursive_cost
    }

    /// Returns the expected memory cost, in bytes, of unitary compression.
    pub const fn unitary_cost(&self) -> u64 {
        self.unitary_cost
    }

    /// Estimates the memory cost of unitary compression.
    ///
    /// The cost is estimated as the number of bytes required to encode all instances in the cluster
    /// in terms of the center of the cluster.
    pub(crate) fn estimate_unitary_cost<I: Instance, D: SquishyDataset<I, U>>(&self, data: &D) -> u64 {
        data.bytes_per_unit_distance() * self.radius().as_u64() * self.cardinality() as u64
    }

    /// Estimates the memory cost of recursive compression, i.e. the cost of compressing the
    /// centers of the children in terms of the center of the cluster.
    pub(crate) fn estimate_recursive_cost<I: Instance, D: SquishyDataset<I, U>>(&self, data: &D) -> u64 {
        match self.children() {
            Some([left, right]) => {
                let left_center = left.arg_center();
                let right_center = right.arg_center();
                let [left, right, center] = [&data[left_center], &data[right_center], &data[self.arg_center()]];
                let left_cost = data
                    .encode_instance(center, left)
                    .unwrap_or_else(|_| unreachable!("We control the instances."))
                    .len()
                    .as_u64();
                let right_cost = data
                    .encode_instance(center, right)
                    .unwrap_or_else(|_| unreachable!("We control the instances."))
                    .len()
                    .as_u64();
                left_cost + right_cost
            }
            None => {
                // If there are no children, the cost is zero.
                0
            }
        }
    }

    /// Recursively estimates and sets the costs of recursive and unitary compression in the subtree.
    pub(crate) fn estimate_costs<I: Instance, D: SquishyDataset<I, U>>(&mut self, data: &D) {
        if let Some(children) = self.children.as_mut() {
            children.left.estimate_costs(data);
            children.right.estimate_costs(data);
            self.recursive_cost = self.estimate_recursive_cost(data);
        } else {
            self.recursive_cost = 0;
        }
        self.unitary_cost = self.estimate_unitary_cost(data);
    }
}

impl<U: Int> Cluster<U> for SquishyBall<U> {
    fn new_root<I: Instance, D: Dataset<I, U>>(data: &D, seed: Option<u64>) -> Self {
        let uni_ball = UniBall::new_root(data, seed);
        Self {
            uni_ball,
            recursive_cost: 0,
            unitary_cost: 0,
            children: None,
        }
    }

    fn partition<I, D, P>(self, data: &mut D, criteria: &P, seed: Option<u64>) -> Self
    where
        I: Instance,
        D: Dataset<I, U>,
        P: PartitionCriterion<U>,
    {
        let uni_ball = self.uni_ball.partition(data, criteria, seed);
        Self::from_base_tree(uni_ball)
    }

    fn offset(&self) -> usize {
        self.uni_ball.offset()
    }

    fn cardinality(&self) -> usize {
        self.uni_ball.cardinality()
    }

    fn depth(&self) -> usize {
        self.uni_ball.depth()
    }

    fn arg_center(&self) -> usize {
        self.uni_ball.arg_center()
    }

    fn radius(&self) -> U {
        self.uni_ball.radius()
    }

    fn arg_radial(&self) -> usize {
        self.uni_ball.arg_radial()
    }

    fn lfd(&self) -> f64 {
        self.uni_ball.lfd()
    }

    fn children(&self) -> Option<[&Self; 2]> {
        self.children.as_ref().map(|c| [c.left.as_ref(), c.right.as_ref()])
    }

    fn polar_distance(&self) -> Option<U> {
        self.uni_ball.polar_distance()
    }

    fn arg_poles(&self) -> Option<[usize; 2]> {
        self.uni_ball.arg_poles()
    }
}

impl<U: Int> PartialEq for SquishyBall<U> {
    fn eq(&self, other: &Self) -> bool {
        self.uni_ball.eq(&other.uni_ball)
    }
}

impl<U: Int> Eq for SquishyBall<U> {}

impl<U: Int> PartialOrd for SquishyBall<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: Int> Ord for SquishyBall<U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.uni_ball.cmp(&other.uni_ball)
    }
}

impl<U: Int> Hash for SquishyBall<U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uni_ball.hash(state);
    }
}

impl<U: Int> Display for SquishyBall<U> {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "{}", self.uni_ball)
    }
}

impl<U: Int> Serialize for SquishyBall<U> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("SquishyBall", 4)?;
        state.serialize_field("uni_ball", &self.uni_ball)?;
        state.serialize_field("recursive_cost", &self.recursive_cost)?;
        state.serialize_field("unitary_cost", &self.unitary_cost)?;
        state.serialize_field("children", &self.children)?;
        state.end()
    }
}

impl<'de, U: Int> Deserialize<'de> for SquishyBall<U> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// The fields in the `SquishyBall` struct.
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            /// The base `UniBall` of the `SquishyBall`.
            UniBall,
            /// The expected memory cost, in bytes, of recursive compression.
            RecursiveCost,
            /// The expected memory cost, in bytes, of unitary compression.
            UnitaryCost,
            /// The children of the `SquishyBall`.
            Children,
        }

        /// The `Visitor` for the `SquishyBall` struct.
        struct SquishyBallVisitor<U: Int>(PhantomData<U>);

        impl<'de, U: Int> Visitor<'de> for SquishyBallVisitor<U> {
            type Value = SquishyBall<U>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("struct SquishyBall")
            }

            fn visit_seq<V: SeqAccess<'de>>(self, mut seq: V) -> Result<Self::Value, V::Error> {
                let uni_ball = seq
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
                Ok(SquishyBall {
                    uni_ball,
                    recursive_cost,
                    unitary_cost,
                    children,
                })
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Self::Value, V::Error> {
                let mut uni_ball = None;
                let mut recursive_cost = None;
                let mut unitary_cost = None;
                let mut children = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::UniBall => {
                            if uni_ball.is_some() {
                                return Err(serde::de::Error::duplicate_field("uni_ball"));
                            }
                            uni_ball = Some(map.next_value()?);
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

                let uni_ball = uni_ball.ok_or_else(|| serde::de::Error::missing_field("uni_ball"))?;
                let recursive_cost = recursive_cost.ok_or_else(|| serde::de::Error::missing_field("recursive_cost"))?;
                let unitary_cost = unitary_cost.ok_or_else(|| serde::de::Error::missing_field("unitary_cost"))?;
                let children = children.ok_or_else(|| serde::de::Error::missing_field("children"))?;

                Ok(SquishyBall {
                    uni_ball,
                    recursive_cost,
                    unitary_cost,
                    children,
                })
            }
        }

        /// The `Field` names.
        const FIELDS: &[&str] = &["uni_ball", "recursive_cost", "unitary_cost", "children"];
        deserializer.deserialize_struct("SquishyBall", FIELDS, SquishyBallVisitor(PhantomData))
    }
}
