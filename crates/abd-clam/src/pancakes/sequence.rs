//! Implementing `Encodable` and `Decodable` for `Sequence`.

use distances::Number;

use crate::msa::{
    ops::{Edit, Edits},
    Sequence,
};

use super::{Decodable, Encodable};

/// This uses the Needleman-Wunsch algorithm to encode strings.
impl<T: Number> Encodable for Sequence<'_, T> {
    fn as_bytes(&self) -> Box<[u8]> {
        self.seq().as_bytes().to_vec().into_boxed_slice()
    }

    fn encode(&self, reference: &Self) -> Box<[u8]> {
        self.aligner().map_or_else(
            || self.as_bytes(),
            |aligner| {
                let table = aligner.dp_table(self, reference);
                let [s_to_r, r_to_s] = aligner.edits(self, reference, &table);

                let s_check = apply_edits(self.seq(), &s_to_r);
                assert_eq!(s_check, reference.seq(), "From {}, s_to_r: {s_to_r:?}", self.seq());

                let r_check = apply_edits(reference.seq(), &r_to_s);
                assert_eq!(r_check, self.seq(), "From {}, r_to_s: {r_to_s:?}", reference.seq());

                serialize_edits(&r_to_s)
            },
        )
    }
}

/// This uses the Needleman-Wunsch algorithm to decode strings.
impl<T: Number> Decodable for Sequence<'_, T> {
    fn from_bytes(bytes: &[u8]) -> Self {
        let seq =
            String::from_utf8(bytes.to_vec()).unwrap_or_else(|e| unreachable!("Could not cast back to string: {e:?}"));
        Self::new(seq, None)
    }

    fn decode(reference: &Self, bytes: &[u8]) -> Self {
        let edits = deserialize_edits(bytes);
        let seq = apply_edits(reference.seq(), &edits);
        Self::new(seq, reference.aligner())
    }
}

/// Applies a set of edits to a reference (unaligned) string to get a target (unaligned) string.
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
pub fn apply_edits(x: &str, edits: &Edits) -> String {
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
