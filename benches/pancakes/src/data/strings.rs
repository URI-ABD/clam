//! A `Sequence` is a string of characters, sued for -omics data.

use abd_clam::{
    musals::{
        ops::{Edit, Edits},
        Aligner,
    },
    pancakes::{Decoder, Encoder, ParDecoder, ParEncoder},
};
use distances::Number;

use crate::metrics::Hamming;

/// A sequence of characters.
///
/// When encoding with the `Hamming` metric, the `Sequence`s are expected to be
/// the same length. If the user does not ensure this, the behavior will be
/// undefined.
///
/// When encoding with the `Levenshtein` metric, the `Sequence`s need not be the
/// same length. We will use the Needleman-Wunsch algorithm to compute the
/// dynamic programming matrix, and will trace back to find the edits needed to
/// transform one `Sequence` into another.
#[derive(Clone, Debug, bitcode::Encode, bitcode::Decode)]
pub struct Sequence(Vec<u8>);

impl PartialEq for Sequence {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl From<&[u8]> for Sequence {
    fn from(v: &[u8]) -> Self {
        Self(v.to_vec())
    }
}

impl From<&Sequence> for Vec<u8> {
    fn from(s: &Sequence) -> Self {
        s.0.clone()
    }
}

impl From<Vec<u8>> for Sequence {
    fn from(v: Vec<u8>) -> Self {
        Self(v)
    }
}

impl From<&[char]> for Sequence {
    fn from(v: &[char]) -> Self {
        Self(v.iter().map(|&c| c as u8).collect())
    }
}

impl From<&str> for Sequence {
    fn from(s: &str) -> Self {
        Self(s.as_bytes().to_vec())
    }
}

impl From<String> for Sequence {
    fn from(s: String) -> Self {
        Self(s.into_bytes())
    }
}

impl AsRef<[u8]> for Sequence {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsRef<str> for Sequence {
    fn as_ref(&self) -> &str {
        core::str::from_utf8(&self.0).unwrap_or_else(|_| unreachable!("Sequence is not valid UTF-8"))
    }
}

impl IntoIterator for Sequence {
    type Item = u8;
    type IntoIter = std::vec::IntoIter<u8>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromIterator<u8> for Sequence {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl Encoder<Sequence> for Hamming {
    fn to_byte_array(&self, item: &Sequence) -> Box<[u8]> {
        item.0.clone().into_boxed_slice()
    }

    fn encode(&self, item: &Sequence, reference: &Sequence) -> Box<[u8]> {
        // TODO: Cast the index down to the smallest possible type.
        let diffs = item
            .0
            .iter()
            .zip(reference.0.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .flat_map(|(i, (&a, _))| {
                let mut bytes: [u8; 9] = [0; 9];
                bytes[..core::mem::size_of::<usize>()].copy_from_slice(&i.to_le_bytes());
                bytes[core::mem::size_of::<usize>()] = a;
                bytes
            })
            .collect::<Vec<_>>();
        diffs.into_boxed_slice()
    }
}

impl ParEncoder<Sequence> for Hamming {}

impl Encoder<Sequence> for Box<dyn Encoder<Sequence>> {
    fn to_byte_array(&self, item: &Sequence) -> Box<[u8]> {
        (**self).to_byte_array(item)
    }

    fn encode(&self, item: &Sequence, reference: &Sequence) -> Box<[u8]> {
        (**self).encode(item, reference)
    }
}

impl Encoder<Sequence> for Box<dyn ParEncoder<Sequence>> {
    fn to_byte_array(&self, item: &Sequence) -> Box<[u8]> {
        (**self).to_byte_array(item)
    }

    fn encode(&self, item: &Sequence, reference: &Sequence) -> Box<[u8]> {
        (**self).encode(item, reference)
    }
}

impl ParEncoder<Sequence> for Box<dyn ParEncoder<Sequence>> {
    fn par_to_byte_array(&self, item: &Sequence) -> Box<[u8]> {
        (**self).par_to_byte_array(item)
    }

    fn par_encode(&self, item: &Sequence, reference: &Sequence) -> Box<[u8]> {
        (**self).par_encode(item, reference)
    }
}

impl Decoder<Sequence> for Hamming {
    fn from_byte_array(&self, bytes: &[u8]) -> Sequence {
        Sequence::from(bytes)
    }

    fn decode(&self, bytes: &[u8], reference: &Sequence) -> Sequence {
        let mut sequence = reference.0.clone();
        for chunk in bytes.chunks_exact(9) {
            let chunk: [u8; 9] = chunk
                .try_into()
                .unwrap_or_else(|_| unreachable!("chunk is not 9 bytes long"));
            let i = <usize as Number>::from_le_bytes(&chunk[..core::mem::size_of::<usize>()]);
            sequence[i] = chunk[core::mem::size_of::<usize>()];
        }
        Sequence::from(sequence)
    }
}

impl ParDecoder<Sequence> for Hamming {}

impl Decoder<Sequence> for Box<dyn Decoder<Sequence>> {
    fn from_byte_array(&self, bytes: &[u8]) -> Sequence {
        (**self).from_byte_array(bytes)
    }

    fn decode(&self, bytes: &[u8], reference: &Sequence) -> Sequence {
        (**self).decode(bytes, reference)
    }
}

impl Decoder<Sequence> for Box<dyn ParDecoder<Sequence>> {
    fn from_byte_array(&self, bytes: &[u8]) -> Sequence {
        (**self).from_byte_array(bytes)
    }

    fn decode(&self, bytes: &[u8], reference: &Sequence) -> Sequence {
        (**self).decode(bytes, reference)
    }
}

impl ParDecoder<Sequence> for Box<dyn ParDecoder<Sequence>> {
    fn par_from_byte_array(&self, bytes: &[u8]) -> Sequence {
        (**self).par_from_byte_array(bytes)
    }

    fn par_decode(&self, bytes: &[u8], reference: &Sequence) -> Sequence {
        (**self).par_decode(bytes, reference)
    }
}

impl<T: Number> Encoder<Sequence> for Aligner<T> {
    fn to_byte_array(&self, item: &Sequence) -> Box<[u8]> {
        item.0.clone().into_boxed_slice()
    }

    fn encode(&self, item: &Sequence, reference: &Sequence) -> Box<[u8]> {
        let table = self.dp_table(item, reference);
        let [_, r_to_i] = self.edits(item, reference, &table);
        serialize_edits(&r_to_i)
    }
}

impl<T: Number> ParEncoder<Sequence> for Aligner<T> {}

impl<T: Number> Decoder<Sequence> for Aligner<T> {
    fn from_byte_array(&self, bytes: &[u8]) -> Sequence {
        Sequence::from(bytes)
    }

    fn decode(&self, bytes: &[u8], reference: &Sequence) -> Sequence {
        let edits = deserialize_edits(bytes);
        Sequence::from(apply_edits(reference.as_ref(), &edits))
    }
}

impl<T: Number> ParDecoder<Sequence> for Aligner<T> {}

/// Applies a set of edits to a reference (unaligned) string to get a target
/// (unaligned) string.
///
/// # Arguments
///
/// * `x`: The unaligned reference string.
/// * `edits`: The edits to apply to the reference string.
///
/// # Returns
///
/// The unaligned target string.
#[must_use]
fn apply_edits(x: &str, edits: &Edits) -> String {
    let mut x: Vec<u8> = x.as_bytes().to_vec();

    for (i, edit) in edits.as_ref() {
        match edit {
            Edit::Sub(c) => {
                x[*i] = *c;
            }
            Edit::Ins(c) => {
                x.insert(*i, *c);
            }
            Edit::Del => {
                x.remove(*i);
            }
        }
    }

    String::from_utf8(x).unwrap_or_else(|e| unreachable!("Could not cast back to string: {e:?}"))
}

/// Serializes a vector of edit operations into a byte array.
fn serialize_edits(edits: &Edits) -> Box<[u8]> {
    let bytes = edits.as_ref().iter().flat_map(edit_to_bin).collect::<Vec<_>>();
    bytes.into_boxed_slice()
}

/// Encodes an edit operation into a byte array.
///
/// A `Del` edit is encoded as `10` followed by the index of the edit in 14 bits.
/// A `Ins` edit is encoded as `01` followed by the index of the edit in 14 bits and the character in 8 bits.
/// A `Sub` edit is encoded as `11` followed by the index of the edit in 14 bits and the character in 8 bits.
///
/// # Arguments
///
/// * `edit`: The edit operation.
///
/// # Returns
///
/// A byte array encoding the edit operation.
#[allow(clippy::cast_possible_truncation)]
fn edit_to_bin((i, edit): &(usize, Edit)) -> Vec<u8> {
    let mask_idx = 0b00_111111;
    let mask_del = 0b10_000000;
    let mask_ins = 0b01_000000;
    let mask_sub = 0b11_000000;

    // First 2 bits for the type of edit, 14 bits for the index.
    let mut bytes = (*i as u16).to_be_bytes().to_vec();
    bytes[0] &= mask_idx;

    match edit {
        Edit::Del => {
            bytes[0] |= mask_del;
        }
        Edit::Ins(c) => {
            bytes[0] |= mask_ins;
            // 8 bits for the character.
            bytes.push(*c);
        }
        Edit::Sub(c) => {
            bytes[0] |= mask_sub;
            // 8 bits for the character.
            bytes.push(*c);
        }
    }
    bytes
}

/// Deserializes a byte array into a vector of edit operations.
///
/// A `Del` edit is encoded as `10` followed by the index of the edit in 14 bits.
/// A `Ins` edit is encoded as `01` followed by the index of the edit in 14 bits and the character in 8 bits.
/// A `Sub` edit is encoded as `11` followed by the index of the edit in 14 bits and the character in 8 bits.
///
/// # Arguments
///
/// * `bytes`: The byte array encoding the edit operations.
///
/// # Errors
///
/// * If the byte array is not a valid encoding of edit operations.
/// * If the edit type is not recognized.
///
/// # Returns
///
/// A vector of edit operations.
fn deserialize_edits(bytes: &[u8]) -> Edits {
    let mut edits = Vec::new();
    let mut offset = 0;
    let mask_idx = 0b00_111111;

    while offset < bytes.len() {
        let edit_bits = bytes[offset] & !mask_idx;
        let i = u16::from_be_bytes([bytes[offset] & mask_idx, bytes[offset + 1]]) as usize;
        let edit = match edit_bits {
            0b10_000000 => {
                offset += 2;
                Edit::Del
            }
            0b01_000000 => {
                let c = bytes[offset + 2];
                offset += 3;
                Edit::Ins(c)
            }
            0b11_000000 => {
                let c = bytes[offset + 2];
                offset += 3;
                Edit::Sub(c)
            }
            _ => unreachable!("Invalid edit type: {edit_bits:b}."),
        };
        edits.push((i, edit));
    }

    Edits::from(edits)
}
