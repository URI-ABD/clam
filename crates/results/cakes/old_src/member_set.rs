//! Representing genomic sequences.

use std::collections::HashSet;

use abd_clam::cakes::{Decodable, Encodable};
use serde::{Deserialize, Serialize};

/// A genomic sequence from an MSA.
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct MemberSet(HashSet<usize>);

impl MemberSet {
    /// Create a new sequence.
    pub fn new(items: &[usize]) -> Self {
        Self(items.iter().copied().collect())
    }

    /// Get the sequence.
    pub const fn inner(&self) -> &HashSet<usize> {
        &self.0
    }

    /// Get the length of the sequence.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the sequence without gaps or padding.
    pub fn as_vec(&self) -> Vec<usize> {
        self.0.iter().copied().collect()
    }
}

impl Encodable for MemberSet {
    fn as_bytes(&self) -> Box<[u8]> {
        self.0
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    fn encode(&self, reference: &Self) -> Box<[u8]> {
        // Get the items that are not in the reference.
        let new_items = self.0.difference(&reference.0).copied().collect::<Vec<_>>();
        // Get the items that are not in self.
        let removed_items = reference.0.difference(&self.0).copied().collect::<Vec<_>>();

        let mut bytes = vec![];
        bytes.extend_from_slice(&new_items.len().to_le_bytes());
        bytes.extend(new_items.iter().flat_map(|i| i.to_le_bytes()));

        bytes.extend_from_slice(&removed_items.len().to_le_bytes());
        bytes.extend(removed_items.iter().flat_map(|i| i.to_le_bytes()));

        bytes.into_boxed_slice()
    }
}

impl Decodable for MemberSet {
    fn from_bytes(bytes: &[u8]) -> Self {
        let items = bytes
            .chunks_exact(core::mem::size_of::<usize>())
            .map(|chunk| usize::from_le_bytes(chunk.try_into().unwrap_or_else(|e| unreachable!("{e}"))))
            .collect::<HashSet<_>>();

        Self(items)
    }

    fn decode(reference: &Self, bytes: &[u8]) -> Self {
        let mut member_set = reference.0.clone();

        let mut offset = 0;

        let num_new_items = abd_clam::cakes::read_usize(bytes, &mut offset);
        for _ in 0..num_new_items {
            member_set.insert(abd_clam::cakes::read_usize(bytes, &mut offset));
        }

        let num_removed_items = abd_clam::cakes::read_usize(bytes, &mut offset);
        for _ in 0..num_removed_items {
            member_set.remove(&abd_clam::cakes::read_usize(bytes, &mut offset));
        }

        Self(member_set)
    }
}
