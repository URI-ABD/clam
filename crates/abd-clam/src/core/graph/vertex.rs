//! A `Vertex` for a `Graph`.

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

/// The ratios used for anomaly detection.
pub type Ratios = [f64; 6];

/// A `Vertex` for a `Graph`.
#[derive(Debug)]
pub struct Vertex<U: Number> {
    /// The base `BaseCluster` of the `Vertex`.
    base_cluster: BaseCluster<U>,
    /// The ratios used for anomaly detection.
    ratios: Ratios,
    /// Child Vertices
    children: Option<Children<U, Self>>,
}

impl<U: Number> Vertex<U> {
    /// Creates a new `Vertex`.
    pub const fn new(base_cluster: BaseCluster<U>, ratios: Ratios, children: Option<Children<U, Self>>) -> Self {
        Self {
            base_cluster,
            ratios,
            children,
        }
    }

    /// Creates a new `Vertex` tree.
    pub const fn from_base_tree(base_cluster: BaseCluster<U>) -> Self {
        let ratios = [0.0; 6];
        // TODO: Calculate the ratios
        Self::new(base_cluster, ratios, None)
    }

    /// Normalizes the ratios in the subtree.
    #[must_use]
    pub fn normalize_ratios(self) -> Self {
        todo!()
    }

    /// The base `BaseCluster` of the `Vertex`.
    pub const fn uni_ball(&self) -> &BaseCluster<U> {
        &self.base_cluster
    }

    /// The ratios of the `Vertex`.
    pub const fn ratios(&self) -> Ratios {
        self.ratios
    }
}

impl<U: Number> Cluster<U> for Vertex<U> {
    fn new_root<I: Instance, D: Dataset<I, U>>(data: &D, seed: Option<u64>) -> Self {
        let uni_ball = BaseCluster::new_root(data, seed);
        let ratios = [0.0; 6];
        Self::new(uni_ball, ratios, None)
    }

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

impl<U: Number> PartialEq for Vertex<U> {
    fn eq(&self, other: &Self) -> bool {
        self.base_cluster.eq(&other.base_cluster)
    }
}

impl<U: Number> Eq for Vertex<U> {}

impl<U: Number> PartialOrd for Vertex<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: Number> Ord for Vertex<U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.base_cluster.cmp(&other.base_cluster)
    }
}

impl<U: Number> Hash for Vertex<U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base_cluster.hash(state);
    }
}

impl<U: Number> Display for Vertex<U> {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "{}", self.base_cluster)
    }
}

impl<U: Number> Serialize for Vertex<U> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Vertex", 3)?;
        state.serialize_field("uni_ball", &self.base_cluster)?;
        state.serialize_field("ratios", &self.ratios)?;
        state.serialize_field("children", &self.children)?;
        state.end()
    }
}

impl<'de, U: Number> Deserialize<'de> for Vertex<U> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// The fields in the `Vertex` struct.
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            /// The base `BaseCluster` of the `Vertex`.
            BaseCluster,
            /// The ratios of the `Vertex`.
            Ratios,
            /// The children of the `Vertex`.
            Children,
        }

        /// The `Visitor` for the `Vertex` struct.
        struct VertexVisitor<U: Number>(PhantomData<U>);

        impl<'de, U: Number> Visitor<'de> for VertexVisitor<U> {
            type Value = Vertex<U>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("struct Vertex")
            }

            fn visit_seq<V: SeqAccess<'de>>(self, mut seq: V) -> Result<Self::Value, V::Error> {
                let uni_ball = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let ratios = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let children = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                Ok(Vertex::new(uni_ball, ratios, children))
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Self::Value, V::Error> {
                let mut uni_ball = None;
                let mut ratios = None;
                let mut children = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::BaseCluster => {
                            if uni_ball.is_some() {
                                return Err(serde::de::Error::duplicate_field("uni_ball"));
                            }
                            uni_ball = Some(map.next_value()?);
                        }
                        Field::Ratios => {
                            if ratios.is_some() {
                                return Err(serde::de::Error::duplicate_field("ratios"));
                            }
                            ratios = Some(map.next_value()?);
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
                let ratios = ratios.ok_or_else(|| serde::de::Error::missing_field("ratios"))?;
                let children = children.ok_or_else(|| serde::de::Error::missing_field("children"))?;

                Ok(Vertex::new(uni_ball, ratios, children))
            }
        }

        /// The `Field` names.
        const FIELDS: &[&str] = &["uni_ball", "ratios"];
        deserializer.deserialize_struct("Vertex", FIELDS, VertexVisitor(PhantomData))
    }
}
