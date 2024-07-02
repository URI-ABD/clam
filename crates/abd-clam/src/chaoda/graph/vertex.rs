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

use crate::{core::cluster::Children, utils, Cluster, Dataset, Instance, PartitionCriterion, UniBall};

/// The ratios used for anomaly detection.
pub type Ratios = [f64; 6];

/// A `Vertex` for a `Graph`.
#[derive(Debug, Clone)]
pub struct Vertex<U: Number> {
    /// The base `UniBall` of the `Vertex`.
    uni_ball: UniBall<U>,
    /// The ratios used for anomaly detection.
    ratios: Ratios,
    /// Child Vertices
    children: Option<Children<U, Self>>,
}

impl<I: Instance, U: Number, D: Dataset<I, U>> crate::Tree<I, U, D, Vertex<U>> {
    /// Sets the `Vertex` ratios for anomaly detection and related applications.
    ///
    /// This should only be called on the root `Cluster` after calling `partition`.
    ///
    /// # Arguments
    ///
    /// * `normalized`: Whether to apply Gaussian error normalization to the ratios.
    #[must_use]
    pub fn normalize_ratios(mut self) -> Self {
        self.root = self.root.normalize_ratios();
        self
    }
}

impl<U: Number> Vertex<U> {
    /// Creates a new `Vertex`.
    pub const fn new(uni_ball: UniBall<U>, ratios: Ratios, children: Option<Children<U, Self>>) -> Self {
        Self {
            uni_ball,
            ratios,
            children,
        }
    }

    /// Creates a new `Vertex` tree.
    pub fn from_base_tree(root: UniBall<U>) -> Self {
        Self::from_uni_ball(root).set_child_parent_ratios([1.0; 6])
    }

    /// Recursively creates a new `Vertex` tree.
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
                Self::new(uni_ball, [1.0; 6], Some(children))
            }
            None => Self::new(uni_ball, [1.0; 6], None),
        }
    }

    /// Set the child-parent ratios.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub(crate) fn set_child_parent_ratios(mut self, parent_ratios: Ratios) -> Self {
        let [pc, pr, pl, pc_, pr_, pl_] = parent_ratios;

        let c = self.cardinality().as_f64() / pc;
        let r = self.radius().as_f64() / pr;
        let l = self.lfd() / pl;

        let c_ = utils::next_ema(c, pc_);
        let r_ = utils::next_ema(r, pr_);
        let l_ = utils::next_ema(l, pl_);

        let ratios = [c, r, l, c_, r_, l_];
        self.ratios = ratios;

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

    /// Normalizes the ratios in the subtree.
    #[must_use]
    pub fn normalize_ratios(mut self) -> Self {
        let all_ratios = self.subtree().into_iter().map(Self::ratios).collect::<Vec<_>>();

        let all_ratios = utils::rows_to_cols(&all_ratios);

        // mean of each column
        let means = utils::calc_row_means(&all_ratios);

        // sd of each column
        let sds = utils::calc_row_sds(&all_ratios);

        self.set_normalized_ratios(means, sds);

        self
    }

    /// Recursively applies Gaussian error normalization to the ratios in the subtree.
    fn set_normalized_ratios(&mut self, means: Ratios, sds: Ratios) {
        let normalized_ratios: Vec<_> = self
            .ratios
            .into_iter()
            .zip(means)
            .zip(sds)
            .map(|((value, mean), std)| (value - mean) / std.mul_add(core::f64::consts::SQRT_2, f64::EPSILON))
            .map(libm::erf)
            .map(|v| (1. + v) / 2.)
            .collect();

        if let Ok(normalized_ratios) = normalized_ratios.try_into() {
            self.ratios = normalized_ratios;
        }

        match &mut self.children {
            Some(children) => {
                children.left.set_normalized_ratios(means, sds);
                children.right.set_normalized_ratios(means, sds);
            }
            None => (),
        }
    }

    /// The base `UniBall` of the `Vertex`.
    pub const fn uni_ball(&self) -> &UniBall<U> {
        &self.uni_ball
    }

    /// The ratios of the `Vertex`.
    pub const fn ratios(&self) -> Ratios {
        self.ratios
    }
}

impl<U: Number> Cluster<U> for Vertex<U> {
    fn new_root<I: Instance, D: Dataset<I, U>>(data: &D, seed: Option<u64>) -> Self {
        let uni_ball = UniBall::new_root(data, seed);
        let ratios = [0.0; 6];
        Self::new(uni_ball, ratios, None)
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

impl<U: Number> PartialEq for Vertex<U> {
    fn eq(&self, other: &Self) -> bool {
        self.uni_ball.eq(&other.uni_ball)
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
        self.uni_ball.cmp(&other.uni_ball)
    }
}

impl<U: Number> Hash for Vertex<U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uni_ball.hash(state);
    }
}

impl<U: Number> Display for Vertex<U> {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "{}", self.uni_ball)
    }
}

impl<U: Number> Serialize for Vertex<U> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Vertex", 3)?;
        state.serialize_field("uni_ball", &self.uni_ball)?;
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
            /// The base `UniBall` of the `Vertex`.
            UniBall,
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
                        Field::UniBall => {
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
        const FIELDS: &[&str] = &["uni_ball", "ratios", "children"];
        deserializer.deserialize_struct("Vertex", FIELDS, VertexVisitor(PhantomData))
    }
}
