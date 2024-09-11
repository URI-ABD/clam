//! Representing genomic sequences.

#![allow(dead_code)]

use abd_clam::cakes::{Decodable, Encodable};
use serde::{Deserialize, Serialize};

/// A genomic sequence from an MSA.
#[derive(Serialize, Deserialize, Default, Clone)]
#[allow(clippy::module_name_repetitions)]
pub struct AlignedSequence(String);

impl AlignedSequence {
    /// Create a new sequence.
    pub const fn new(sequence: String) -> Self {
        Self(sequence)
    }

    /// Get the sequence.
    pub fn sequence(&self) -> &str {
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
    pub fn as_unaligned(&self) -> String {
        self.0.chars().filter(|&c| c != '-' && c != '.').collect()
    }
}

impl Encodable for AlignedSequence {
    fn as_bytes(&self) -> Box<[u8]> {
        self.0.as_bytes().to_vec().into_boxed_slice()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode(&self, reference: &Self) -> Box<[u8]> {
        // We assume that the sequences are all the same length so we can just encode the differences.
        self.0
            .chars()
            .zip(reference.0.chars())
            .enumerate()
            .filter_map(|(i, (c, r))| if c == r { None } else { Some((i as u16, c)) })
            .flat_map(|(i, c)| {
                let mut bytes = vec![];
                bytes.extend_from_slice(&i.to_be_bytes());
                bytes.push(c as u8);
                bytes
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

impl Decodable for AlignedSequence {
    fn from_bytes(bytes: &[u8]) -> Self {
        Self(String::from_utf8(bytes.to_vec()).unwrap_or_else(|e| unreachable!("Error decoding sequence: {e}")))
    }

    fn decode(reference: &Self, bytes: &[u8]) -> Self {
        let mut sequence = reference.0.as_bytes().to_owned();

        for chunk in bytes.chunks_exact(3) {
            let i = u16::from_be_bytes([chunk[0], chunk[1]]) as usize;
            sequence[i] = chunk[2];
        }

        Self::from_bytes(&sequence)
    }
}
