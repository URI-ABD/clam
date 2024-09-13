//! Aligned sequence with Hamming distance and substitutions for Edits.

use abd_clam::cakes::{Decodable, Encodable};
use distances::number::UInt;
use serde::{Deserialize, Serialize};

/// A sequence from a FASTA file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aligned<U: UInt> {
    /// The aligned sequence.
    seq: String,
    /// To keep the type parameter.
    _phantom: std::marker::PhantomData<U>,
}

impl<U: UInt> Default for Aligned<U> {
    fn default() -> Self {
        Self {
            seq: String::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<U: UInt> Aligned<U> {
    /// Returns the Hamming metric for `Aligned` sequences.
    #[must_use]
    pub fn metric() -> abd_clam::Metric<Self, U> {
        let distance_function = |first: &Self, second: &Self| {
            U::from(
                first
                    .seq
                    .chars()
                    .zip(second.seq.chars())
                    .filter(|(a, b)| a != b)
                    .count(),
            )
        };
        abd_clam::Metric::new(distance_function, false)
    }

    /// Returns the Levenshtein metric for unaligned version of the `Aligned` sequences.
    #[must_use]
    pub fn levenshtein_metric() -> abd_clam::Metric<Self, U> {
        let distance_function = |first: &Self, second: &Self| {
            U::from(stringzilla::sz::edit_distance(
                first.as_unaligned(),
                second.as_unaligned(),
            ))
        };
        abd_clam::Metric::new(distance_function, false)
    }

    pub fn as_unaligned(&self) -> String {
        self.seq.chars().filter(|&c| c != '-' && c != '.').collect()
    }
}

impl<U: UInt> AsRef<str> for Aligned<U> {
    fn as_ref(&self) -> &str {
        &self.seq
    }
}

impl<U: UInt> From<&str> for Aligned<U> {
    fn from(seq: &str) -> Self {
        Self::from(seq.to_string())
    }
}

impl<U: UInt> From<String> for Aligned<U> {
    fn from(seq: String) -> Self {
        Self {
            seq,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<U: UInt> Encodable for Aligned<U> {
    fn as_bytes(&self) -> Box<[u8]> {
        self.seq.as_bytes().into()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode(&self, reference: &Self) -> Box<[u8]> {
        self.seq
            .chars()
            .zip(reference.seq.chars())
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

/// This uses the Needleman-Wunsch algorithm to decode strings.
impl<U: UInt> Decodable for Aligned<U> {
    fn from_bytes(bytes: &[u8]) -> Self {
        Self::from(
            String::from_utf8(bytes.to_vec()).unwrap_or_else(|e| unreachable!("Could not cast back to string: {e:?}")),
        )
    }

    fn decode(reference: &Self, bytes: &[u8]) -> Self {
        let mut sequence = reference.seq.chars().collect::<Vec<_>>();

        for chunk in bytes.chunks_exact(3) {
            let i = u16::from_be_bytes([chunk[0], chunk[1]]) as usize;
            let c = chunk[2] as char;
            sequence[i] = c;
        }

        Self::from(sequence.into_iter().collect::<String>())
    }
}
