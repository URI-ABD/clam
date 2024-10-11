//! A `HashSet` of named members under Jaccard distance and set difference for encoding.

use core::hash::Hash;

use std::collections::HashSet;

use abd_clam::cakes::{Decodable, Encodable};
use distances::number::{Float, UInt};
use serde::{Deserialize, Serialize};

/// A set of named members.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberSet<T: Hash + Eq + Copy, F: Float> {
    /// The members of the set.
    inner: HashSet<T>,
    /// To keep the type parameter.
    _phantom: std::marker::PhantomData<F>,
}

impl<T: Hash + Eq + Copy, F: Float> Default for MemberSet<T, F> {
    fn default() -> Self {
        Self {
            inner: HashSet::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Hash + Eq + Copy, F: Float> MemberSet<T, F> {
    /// Returns the Jaccard distance metric for `MemberSet`s.
    #[must_use]
    pub fn metric() -> abd_clam::Metric<Self, F> {
        let distance_function = |first: &Self, second: &Self| {
            let intersection = first.inner.intersection(&second.inner).count();
            let union = first.inner.len() + second.inner.len() - intersection;
            let sim = if union == 0 {
                F::ZERO
            } else {
                F::from(intersection) / F::from(union)
            };
            F::ONE - sim
        };

        abd_clam::Metric::new(distance_function, false)
    }
}

impl<T: Hash + Eq + Copy, F: Float> From<&[T]> for MemberSet<T, F> {
    fn from(items: &[T]) -> Self {
        Self {
            inner: items.iter().copied().collect(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Hash + Eq + Copy, F: Float> From<&Vec<T>> for MemberSet<T, F> {
    fn from(items: &Vec<T>) -> Self {
        Self::from(items.as_slice())
    }
}

impl<T: Hash + Eq + Copy, F: Float> From<&HashSet<T>> for MemberSet<T, F> {
    fn from(items: &HashSet<T>) -> Self {
        Self {
            inner: items.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Hash + Eq + Copy, F: Float> AsRef<HashSet<T>> for MemberSet<T, F> {
    fn as_ref(&self) -> &HashSet<T> {
        &self.inner
    }
}

impl<T: Hash + Eq + Copy, F: Float> From<&MemberSet<T, F>> for Vec<T> {
    fn from(set: &MemberSet<T, F>) -> Self {
        set.inner.iter().copied().collect()
    }
}

impl<T: UInt, F: Float> Encodable for MemberSet<T, F> {
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

        let new_items = self.inner.difference(&reference.inner).copied().collect::<Vec<_>>();
        bytes.extend_from_slice(&new_items.len().to_le_bytes());
        bytes.extend(new_items.into_iter().flat_map(T::to_le_bytes));

        let removed_items = reference.inner.difference(&self.inner).copied().collect::<Vec<_>>();
        bytes.extend_from_slice(&removed_items.len().to_le_bytes());
        bytes.extend(removed_items.into_iter().flat_map(T::to_le_bytes));

        bytes.into_boxed_slice()
    }
}

impl<T: UInt, F: Float> Decodable for MemberSet<T, F> {
    fn from_bytes(bytes: &[u8]) -> Self {
        let items = bytes
            .chunks_exact(T::NUM_BYTES)
            .map(T::from_le_bytes)
            .collect::<HashSet<_>>();

        Self {
            inner: items,
            _phantom: std::marker::PhantomData,
        }
    }

    fn decode(reference: &Self, bytes: &[u8]) -> Self {
        let mut inner = reference.inner.clone();

        let mut offset = 0;

        let num_new_items = abd_clam::utils::read_number::<usize>(bytes, &mut offset);
        for _ in 0..num_new_items {
            inner.insert(abd_clam::utils::read_number(bytes, &mut offset));
        }

        let num_removed_items = abd_clam::utils::read_number::<usize>(bytes, &mut offset);
        for _ in 0..num_removed_items {
            inner.remove(&abd_clam::utils::read_number(bytes, &mut offset));
        }

        Self {
            inner,
            _phantom: std::marker::PhantomData,
        }
    }
}
