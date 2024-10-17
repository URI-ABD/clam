//! Unaligned sequence with Levenshtein distance and Needleman-Wunsch Edits.

use abd_clam::{
    msa::Sequence,
    pancakes::{Decodable, Encodable},
};
use distances::Number;
use serde::{Deserialize, Serialize};

/// A sequence from a FASTA file.
#[derive(Debug, Clone, Serialize, Deserialize, Default, bitcode::Encode, bitcode::Decode)]
pub struct Unaligned(String);

impl AsRef<str> for Unaligned {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl AsRef<[u8]> for Unaligned {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl From<&str> for Unaligned {
    fn from(seq: &str) -> Self {
        Self(seq.chars().filter(|&c| c != '-' && c != '.').collect())
    }
}

impl From<String> for Unaligned {
    fn from(seq: String) -> Self {
        Self::from(seq.as_str())
    }
}

impl<'a, T: Number> From<Sequence<'a, T>> for Unaligned {
    fn from(seq: Sequence<'a, T>) -> Self {
        Self::from(seq.as_ref())
    }
}

impl From<Unaligned> for String {
    fn from(seq: Unaligned) -> Self {
        seq.0
    }
}

impl Encodable for Unaligned {
    fn as_bytes(&self) -> Box<[u8]> {
        self.0.as_bytes().into()
    }

    fn encode(&self, reference: &Self) -> Box<[u8]> {
        let (x, y) = (self.as_ref(), reference.as_ref());

        let penalties = distances::strings::Penalties::default();
        let table = distances::strings::needleman_wunsch::compute_table::<u32>(x, y, penalties);

        #[allow(clippy::tuple_array_conversions)]
        let (x, y) = distances::strings::needleman_wunsch::trace_back_recursive(&table, [x, y]);
        let edits = distances::strings::needleman_wunsch::unaligned_x_to_y(&x, &y);
        serialize_edits(&edits)
    }
}

/// This uses the Needleman-Wunsch algorithm to decode strings.
impl Decodable for Unaligned {
    fn from_bytes(bytes: &[u8]) -> Self {
        let seq =
            String::from_utf8(bytes.to_vec()).unwrap_or_else(|e| unreachable!("Could not cast back to string: {e:?}"));
        Self::from(seq)
    }

    fn decode(reference: &Self, bytes: &[u8]) -> Self {
        let edits = deserialize_edits(bytes);
        let seq = distances::strings::needleman_wunsch::apply_edits(reference.as_ref(), &edits);
        Self::from(seq)
    }
}

/// Serializes a vector of edit operations into a byte array.
fn serialize_edits(edits: &[distances::strings::Edit]) -> Box<[u8]> {
    let bytes = edits.iter().flat_map(edit_to_bin).collect::<Vec<_>>();
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
fn edit_to_bin(edit: &distances::strings::Edit) -> Vec<u8> {
    let mask_idx = 0b00_111111;
    let mask_del = 0b10_000000;
    let mask_ins = 0b01_000000;
    let mask_sub = 0b11_000000;

    match edit {
        distances::strings::Edit::Del(i) => {
            let mut bytes = (*i as u16).to_be_bytes().to_vec();
            // First 2 bits for the type of edit, 14 bits for the index.
            bytes[0] &= mask_idx;
            bytes[0] |= mask_del;
            bytes
        }
        distances::strings::Edit::Ins(i, c) => {
            let mut bytes = (*i as u16).to_be_bytes().to_vec();
            // First 2 bits for the type of edit, 14 bits for the index.
            bytes[0] &= mask_idx;
            bytes[0] |= mask_ins;
            // 8 bits for the character.
            bytes.push(*c as u8);
            bytes
        }
        distances::strings::Edit::Sub(i, c) => {
            let mut bytes = (*i as u16).to_be_bytes().to_vec();
            // First 2 bits for the type of edit, 14 bits for the index.
            bytes[0] &= mask_idx;
            bytes[0] |= mask_sub;
            // 8 bits for the character.
            bytes.push(*c as u8);
            bytes
        }
    }
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
fn deserialize_edits(bytes: &[u8]) -> Vec<distances::strings::Edit> {
    let mut edits = Vec::new();
    let mut offset = 0;
    let mask_idx = 0b00_111111;

    while offset < bytes.len() {
        let edit_bits = bytes[offset] & !mask_idx;
        let i = u16::from_be_bytes([bytes[offset] & mask_idx, bytes[offset + 1]]) as usize;
        let edit = match edit_bits {
            0b10_000000 => {
                offset += 2;
                distances::strings::Edit::Del(i)
            }
            0b01_000000 => {
                let c = bytes[offset + 2] as char;
                offset += 3;
                distances::strings::Edit::Ins(i, c)
            }
            0b11_000000 => {
                let c = bytes[offset + 2] as char;
                offset += 3;
                distances::strings::Edit::Sub(i, c)
            }
            _ => unreachable!("Invalid edit type: {edit_bits:b}."),
        };
        edits.push(edit);
    }
    edits
}
