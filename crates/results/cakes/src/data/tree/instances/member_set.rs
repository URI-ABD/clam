//! A `HashSet` of named members under Jaccard distance and set difference for encoding.

use core::hash::Hash;

use std::collections::HashSet;

use abd_clam::{
    metric::ParMetric,
    pancakes::{Decodable, Encodable},
    Metric,
};
use distances::number::{Float, UInt};
use serde::{Deserialize, Serialize};

/// A set of named members.
#[derive(Debug, Clone, Serialize, Deserialize, Default, bitcode::Encode, bitcode::Decode)]
pub struct MemberSet<T: Hash + Eq + Copy>(HashSet<T>);

impl<T: Hash + Eq + Copy> From<&[T]> for MemberSet<T> {
    fn from(items: &[T]) -> Self {
        Self(items.iter().copied().collect())
    }
}

impl<T: Hash + Eq + Copy> From<&Vec<T>> for MemberSet<T> {
    fn from(items: &Vec<T>) -> Self {
        Self::from(items.as_slice())
    }
}

impl<T: Hash + Eq + Copy> From<&HashSet<T>> for MemberSet<T> {
    fn from(items: &HashSet<T>) -> Self {
        Self(items.clone())
    }
}

impl<T: Hash + Eq + Copy> AsRef<HashSet<T>> for MemberSet<T> {
    fn as_ref(&self) -> &HashSet<T> {
        &self.0
    }
}

impl<T: Hash + Eq + Copy> From<&MemberSet<T>> for Vec<T> {
    fn from(set: &MemberSet<T>) -> Self {
        set.0.iter().copied().collect()
    }
}

impl<T: UInt> Encodable for MemberSet<T> {
    fn as_bytes(&self) -> Box<[u8]> {
        Vec::from(self)
            .into_iter()
            .flat_map(T::to_le_bytes)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode(&self, reference: &Self) -> Box<[u8]> {
        let mut bytes = vec![];

        let new_items = self.0.difference(&self.0).copied().collect::<Vec<_>>();
        bytes.extend_from_slice(&new_items.len().to_le_bytes());
        bytes.extend(new_items.into_iter().flat_map(T::to_le_bytes));

        let removed_items = reference.0.difference(&self.0).copied().collect::<Vec<_>>();
        bytes.extend_from_slice(&removed_items.len().to_le_bytes());
        bytes.extend(removed_items.into_iter().flat_map(T::to_le_bytes));

        bytes.into_boxed_slice()
    }
}

impl<T: UInt> Decodable for MemberSet<T> {
    fn from_bytes(bytes: &[u8]) -> Self {
        let items = bytes.chunks_exact(T::NUM_BYTES).map(T::from_le_bytes).collect();
        Self(items)
    }

    fn decode(reference: &Self, bytes: &[u8]) -> Self {
        let mut inner = reference.0.clone();

        let mut offset = 0;

        let num_new_items = abd_clam::utils::read_number::<usize>(bytes, &mut offset);
        for _ in 0..num_new_items {
            inner.insert(abd_clam::utils::read_number(bytes, &mut offset));
        }

        let num_removed_items = abd_clam::utils::read_number::<usize>(bytes, &mut offset);
        for _ in 0..num_removed_items {
            inner.remove(&abd_clam::utils::read_number(bytes, &mut offset));
        }

        Self(inner)
    }
}

/// The `Jaccard` distance metric.
pub struct Jaccard;

impl<T: Hash + Eq + Copy, U: Float> Metric<MemberSet<T>, U> for Jaccard {
    fn distance(&self, a: &MemberSet<T>, b: &MemberSet<T>) -> U {
        let intersection = a.0.intersection(&b.0).count();
        let union = a.0.len() + b.0.len() - intersection;
        if union == 0 {
            U::ZERO
        } else {
            let sim = U::from(intersection) / U::from(union);
            U::ONE - sim
        }
    }

    fn name(&self) -> &str {
        "euclidean"
    }

    fn has_identity(&self) -> bool {
        true
    }

    fn has_non_negativity(&self) -> bool {
        true
    }

    fn has_symmetry(&self) -> bool {
        true
    }

    fn obeys_triangle_inequality(&self) -> bool {
        true
    }

    fn is_expensive(&self) -> bool {
        false
    }
}

impl<T: Hash + Eq + Copy + Send + Sync, U: Float> ParMetric<MemberSet<T>, U> for Jaccard {}
