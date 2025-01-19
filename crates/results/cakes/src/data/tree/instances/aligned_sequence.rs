//! Aligned sequence with Hamming distance and substitutions for Edits.

use abd_clam::{
    metric::ParMetric,
    msa::Sequence,
    pancakes::{Decodable, Encodable},
    Metric,
};
use distances::{number::UInt, Number};
use serde::{Deserialize, Serialize};

/// A sequence from a FASTA file.
#[derive(Debug, Clone, Serialize, Deserialize, Default, bitcode::Encode, bitcode::Decode)]
pub struct Aligned(String);

impl AsRef<str> for Aligned {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Aligned {
    fn from(seq: &str) -> Self {
        Self::from(seq.to_string())
    }
}

impl From<String> for Aligned {
    fn from(seq: String) -> Self {
        Self(seq)
    }
}

impl<'a, T: Number> From<Sequence<'a, T>> for Aligned {
    fn from(seq: Sequence<'a, T>) -> Self {
        Self(seq.seq().to_string())
    }
}

impl Encodable for Aligned {
    fn as_bytes(&self) -> Box<[u8]> {
        self.0.as_bytes().into()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode(&self, reference: &Self) -> Box<[u8]> {
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

/// This uses the Needleman-Wunsch algorithm to decode strings.
impl Decodable for Aligned {
    fn from_bytes(bytes: &[u8]) -> Self {
        Self::from(
            String::from_utf8(bytes.to_vec()).unwrap_or_else(|e| unreachable!("Could not cast back to string: {e:?}")),
        )
    }

    fn decode(reference: &Self, bytes: &[u8]) -> Self {
        let mut sequence = reference.0.chars().collect::<Vec<_>>();

        for chunk in bytes.chunks_exact(3) {
            let i = u16::from_be_bytes([chunk[0], chunk[1]]) as usize;
            let c = chunk[2] as char;
            sequence[i] = c;
        }

        Self::from(sequence.into_iter().collect::<String>())
    }
}

/// The `Hamming` distance metric.
pub struct Hamming;

impl<I: AsRef<str>, U: UInt> Metric<I, U> for Hamming {
    fn distance(&self, a: &I, b: &I) -> U {
        distances::vectors::hamming(a.as_ref().as_ref(), b.as_ref().as_ref())
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

impl<I: AsRef<str> + Send + Sync, U: UInt> ParMetric<I, U> for Hamming {}
