//! A `SquishyBall` is a `Cluster` that supports compression.

use core::{
    cmp::Ordering,
    fmt::{Display, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use distances::{number::UInt, Number};
use rayon::prelude::*;
use serde::{
    de::{MapAccess, SeqAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{core::cluster::Children, Cluster, Dataset, Instance, PartitionCriterion, UniBall};

/// A `SquishyBall` is a `Cluster` that supports compression.
#[derive(Debug, Clone)]
pub struct SquishyBall<U: UInt> {
    /// The `UniBall` for the underlying `Cluster`.
    uni_ball: UniBall<U>,
    /// Expected memory cost, in bytes, of recursive compression.
    recursive_cost: u64,
    /// Expected memory cost, in bytes, of unitary compression.
    unitary_cost: u64,
    /// The minimum expected memory cost, in bytes, of compression.
    min_cost: u64,
    /// Child Clusters
    children: Option<Children<U, Self>>,
    /// Whether to save this cluster to disk or proceed to the next level of the tree.
    squish: bool,
    /// Offset in the compressed data.
    codec_offset: Option<usize>,
}

impl<U: UInt> SquishyBall<U> {
    /// Creates a new `SquishyBall` tree.
    pub fn from_base_tree<I: Instance, D: Dataset<I, U>>(root: UniBall<U>, data: &D) -> Self {
        Self::from_uni_ball(root, data)
    }

    /// Recursively creates a new `SquishyBall` tree.
    fn from_uni_ball<I: Instance, D: Dataset<I, U>>(mut uni_ball: UniBall<U>, data: &D) -> Self {
        let unitary_cost = Self::calculate_unitary_cost(&uni_ball, data);
        match uni_ball.children {
            Some(children) => {
                uni_ball.children = None;
                let (left, right) = rayon::join(
                    || Box::new(Self::from_uni_ball(*children.left, data)),
                    || Box::new(Self::from_uni_ball(*children.right, data)),
                );

                let recursive_cost = {
                    // TODO: Incorporate the `bytes_per_unit_distance` into the cost calculation.
                    let [l_center, r_center, c_center] = [
                        &data[left.arg_center()],
                        &data[right.arg_center()],
                        &data[uni_ball.arg_center()],
                    ];
                    let (l_cost, r_cost) = rayon::join(
                        || Number::as_u64(data.metric()(c_center, l_center)),
                        || Number::as_u64(data.metric()(c_center, r_center)),
                    );
                    l_cost + left.min_cost + r_cost + right.min_cost
                };

                let (squish, min_cost) = if unitary_cost <= recursive_cost {
                    (true, unitary_cost)
                } else {
                    (false, recursive_cost)
                };

                let children = Children {
                    left,
                    right,
                    arg_l: children.arg_l,
                    arg_r: children.arg_r,
                    polar_distance: children.polar_distance,
                };

                Self {
                    uni_ball,
                    recursive_cost,
                    unitary_cost,
                    min_cost,
                    children: Some(children),
                    squish,
                    codec_offset: None,
                }
            }
            None => Self {
                uni_ball,
                recursive_cost: 0,
                unitary_cost,
                min_cost: unitary_cost,
                children: None,
                squish: true,
                codec_offset: None,
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

    /// Returns the lowest expected memory cost, in bytes, of compression.
    /// This is the minimum of the recursive and unitary costs.
    pub const fn min_cost(&self) -> u64 {
        self.min_cost
    }

    /// Returns whether to save this cluster to disk or proceed to the next level of the tree.
    pub const fn squish(&self) -> bool {
        self.squish
    }

    /// Returns the offset in the compressed data.
    pub const fn codec_offset(&self) -> Option<usize> {
        self.codec_offset
    }

    /// Sets the offset in the compressed data.
    pub fn set_codec_offset(&mut self, offset: usize) {
        self.codec_offset = Some(offset);
    }

    /// Returns the compressible subtree.
    pub fn compressible_subtree(&self) -> Vec<&Self> {
        let mut clusters = vec![self];
        if !self.squish {
            if let Some(children) = self.children.as_ref() {
                // If the cluster has children, recursively check the children.
                clusters.extend(children.left.compressible_subtree());
                clusters.extend(children.right.compressible_subtree());
            }
        }
        clusters
    }

    /// Returns the clusters in the subtree that have been marked for squishing.
    pub fn compressible_leaves(&self) -> Vec<&Self> {
        let mut clusters = Vec::new();
        if self.squish {
            // If the cluster is marked for squishing, add it to the list.
            // `squish` is true for leaves, by construction.
            clusters.push(self);
        } else if let Some(children) = self.children.as_ref() {
            // If the cluster has children, recursively check the children.
            clusters.extend(children.left.compressible_leaves());
            clusters.extend(children.right.compressible_leaves());
        }
        clusters
    }

    /// Returns the clusters in the subtree that have been marked for squishing.
    pub fn compressible_leaves_mut(&mut self) -> Vec<&mut Self> {
        let mut clusters = Vec::new();
        if self.squish {
            // If the cluster is marked for squishing, add it to the list.
            // `squish` is true for leaves, by construction.
            clusters.push(self);
        } else if let Some(children) = self.children.as_mut() {
            // If the cluster has children, recursively check the children.
            clusters.extend(children.left.as_mut().compressible_leaves_mut());
            clusters.extend(children.right.as_mut().compressible_leaves_mut());
        }
        clusters
    }

    /// Estimates the memory cost of unitary compression.
    ///
    /// The cost is estimated as the sum of distances from the center to all instances in the cluster.
    fn calculate_unitary_cost<I: Instance, D: Dataset<I, U>>(c: &UniBall<U>, data: &D) -> u64 {
        // TODO: Incorporate the `bytes_per_unit_distance` into the cost calculation.
        let center = &data[c.arg_center()];
        let instances = c.indices().into_par_iter().map(|i| &data[i]);
        let distances = instances.map(|i| data.metric()(center, i)).map(Number::as_u64);
        distances.sum()
    }

    /// Trim the tree by removing the children of those clusters that are marked for squishing.
    pub fn trim(&mut self) {
        if self.squish {
            self.children = None;
        } else if let Some(children) = self.children.as_mut() {
            rayon::join(|| children.left.trim(), || children.right.trim());
        }
    }
}

impl<U: UInt> Cluster<U> for SquishyBall<U> {
    fn new_root<I: Instance, D: Dataset<I, U>>(data: &D, seed: Option<u64>) -> Self {
        let uni_ball = UniBall::new_root(data, seed);
        Self {
            uni_ball,
            recursive_cost: 0,
            unitary_cost: 0,
            min_cost: 0,
            children: None,
            squish: false,
            codec_offset: None,
        }
    }

    fn partition<I, D, P>(self, data: &mut D, criteria: &P, seed: Option<u64>) -> Self
    where
        I: Instance,
        D: Dataset<I, U>,
        P: PartitionCriterion<U>,
    {
        let uni_ball = self.uni_ball.partition(data, criteria, seed);
        Self::from_base_tree(uni_ball, data)
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

impl<U: UInt> PartialEq for SquishyBall<U> {
    fn eq(&self, other: &Self) -> bool {
        self.uni_ball.eq(&other.uni_ball)
    }
}

impl<U: UInt> Eq for SquishyBall<U> {}

impl<U: UInt> PartialOrd for SquishyBall<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: UInt> Ord for SquishyBall<U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.uni_ball.cmp(&other.uni_ball)
    }
}

impl<U: UInt> Hash for SquishyBall<U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uni_ball.hash(state);
    }
}

impl<U: UInt> Display for SquishyBall<U> {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "{}", self.uni_ball)
    }
}

impl<U: UInt> Serialize for SquishyBall<U> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("SquishyBall", 7)?;
        state.serialize_field("uni_ball", &self.uni_ball)?;
        state.serialize_field("recursive_cost", &self.recursive_cost)?;
        state.serialize_field("unitary_cost", &self.unitary_cost)?;
        state.serialize_field("min_cost", &self.min_cost)?;
        state.serialize_field("children", &self.children)?;
        state.serialize_field("squish", &self.squish)?;
        state.serialize_field("codec_offset", &self.codec_offset)?;
        state.end()
    }
}

impl<'de, U: UInt> Deserialize<'de> for SquishyBall<U> {
    #[allow(clippy::too_many_lines)]
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
            /// The minimum expected memory cost, in bytes, of compression.
            MinCost,
            /// The children of the `SquishyBall`.
            Children,
            /// Whether to save this cluster to disk or proceed to the next level of the tree.
            Squish,
            /// The offset in the compressed data.
            CodecOffset,
        }

        /// The `Visitor` for the `SquishyBall` struct.
        struct SquishyBallVisitor<U: UInt>(PhantomData<U>);

        impl<'de, U: UInt> Visitor<'de> for SquishyBallVisitor<U> {
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
                let min_cost = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;
                let children = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(4, &self))?;
                let squish = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(5, &self))?;
                let codec_offset = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(6, &self))?;

                Ok(SquishyBall {
                    uni_ball,
                    recursive_cost,
                    unitary_cost,
                    min_cost,
                    children,
                    squish,
                    codec_offset,
                })
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Self::Value, V::Error> {
                let mut uni_ball = None;
                let mut recursive_cost = None;
                let mut unitary_cost = None;
                let mut min_cost = None;
                let mut children = None;
                let mut squish = None;
                let mut codec_offset = None;

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
                        Field::MinCost => {
                            if min_cost.is_some() {
                                return Err(serde::de::Error::duplicate_field("min_cost"));
                            }
                            min_cost = Some(map.next_value()?);
                        }
                        Field::Children => {
                            if children.is_some() {
                                return Err(serde::de::Error::duplicate_field("children"));
                            }
                            children = Some(map.next_value()?);
                        }
                        Field::Squish => {
                            if squish.is_some() {
                                return Err(serde::de::Error::duplicate_field("squish"));
                            }
                            squish = Some(map.next_value()?);
                        }
                        Field::CodecOffset => {
                            if codec_offset.is_some() {
                                return Err(serde::de::Error::duplicate_field("codec_offset"));
                            }
                            codec_offset = Some(map.next_value()?);
                        }
                    }
                }

                let uni_ball = uni_ball.ok_or_else(|| serde::de::Error::missing_field("uni_ball"))?;
                let recursive_cost = recursive_cost.ok_or_else(|| serde::de::Error::missing_field("recursive_cost"))?;
                let unitary_cost = unitary_cost.ok_or_else(|| serde::de::Error::missing_field("unitary_cost"))?;
                let min_cost = min_cost.ok_or_else(|| serde::de::Error::missing_field("min_cost"))?;
                let children = children.ok_or_else(|| serde::de::Error::missing_field("children"))?;
                let squish = squish.ok_or_else(|| serde::de::Error::missing_field("squish"))?;
                let codec_offset = codec_offset.ok_or_else(|| serde::de::Error::missing_field("codec_offset"))?;

                Ok(SquishyBall {
                    uni_ball,
                    recursive_cost,
                    unitary_cost,
                    min_cost,
                    children,
                    squish,
                    codec_offset,
                })
            }
        }

        /// The `Field` names.
        const FIELDS: &[&str] = &[
            "uni_ball",
            "recursive_cost",
            "unitary_cost",
            "min_cost",
            "children",
            "squish",
            "codec_offset",
        ];
        deserializer.deserialize_struct("SquishyBall", FIELDS, SquishyBallVisitor(PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use distances::strings::levenshtein;

    use crate::{
        pancakes::{decode_general, encode_general, CodecData},
        PartitionCriteria, VecDataset,
    };

    use super::*;

    fn lev_metric(x: &String, y: &String) -> u16 {
        levenshtein(x, y)
    }

    #[test]
    fn test_squishy() -> Result<(), String> {
        let strings = vec![
            "NAJIBPEPPERS-EATS".to_string(),
            "NAJIB-PEPPERSEATS".to_string(),
            "NAJIB-EATSPEPPERS".to_string(),
            "NAJIBEATS-PEPPERS".to_string(),
            "TOM-EATSWHATFOODEATS".to_string(),
            "TOMEATSWHATFOOD-EATS".to_string(),
            "FOODEATS-WHATTOMEATS".to_string(),
            "FOODEATSWHAT-TOMEATS".to_string(),
        ];

        let mut dataset = VecDataset::new("test-genomic".to_string(), strings.clone(), lev_metric, true);
        let criteria = PartitionCriteria::default();
        let seed = Some(42);
        let root = SquishyBall::new_root(&dataset, None).partition(&mut dataset, &criteria, seed);

        let metadata = dataset.metadata().to_vec();
        let dataset = CodecData::new(root, &dataset, encode_general::<u16>, decode_general, metadata)?;

        let clusters = dataset.root().subtree();
        let unitary_costs = clusters.iter().map(|c| c.unitary_cost).collect::<Vec<_>>();
        let recursive_costs = clusters.iter().map(|c| c.recursive_cost).collect::<Vec<_>>();
        let min_costs = clusters.iter().map(|c| c.min_cost).collect::<Vec<_>>();

        for (c, (&u, (&r, &m))) in clusters
            .iter()
            .zip(unitary_costs.iter().zip(recursive_costs.iter().zip(min_costs.iter())))
        {
            assert!(m <= r, "min_cost: {m} > recursive_cost: {r} for cluster: {}", c.name());
            assert!(m <= u, "min_cost: {m} > unitary_cost: {u} for cluster: {}", c.name());
            assert!(
                r <= u,
                "recursive_cost: {r} > unitary_cost: {u} for cluster: {}",
                c.name()
            );
        }

        // assert_eq!(unitary_costs, recursive_costs); // This is just for a sanity check. It is not always true.

        Ok(())
    }
}
