//! Provides Compression and Decompression for Pancakes.

mod cluster;
mod dataset;

pub use cluster::SquishyBall;
#[allow(clippy::module_name_repetitions)]
pub use dataset::CodecData;
use distances::{
    number::UInt,
    strings::{
        needleman_wunsch::{apply_edits, compute_table, trace_back_recursive, unaligned_x_to_y, Edit},
        Penalties,
    },
};

/// A function that encodes a `Instance` into a `Box<[u8]>`.
pub type EncoderFn<I> = fn(&I, &I) -> Result<Box<[u8]>, String>;

/// A function that decodes a `Instance` from a `&[u8]`.
pub type DecoderFn<I> = fn(&I, &[u8]) -> Result<I, String>;

/// Encodes a reference and target string into a byte array.
///
/// # Arguments
///
/// * `reference`: The reference string.
/// * `target`: The target string.
///
/// # Errors
///
/// * If the byte array cannot be serialized.
///
/// # Returns
///
/// A byte array encoding the reference and target strings.
#[allow(dead_code, clippy::ptr_arg)]
pub fn encode_general<U: UInt>(reference: &String, target: &String) -> Result<Box<[u8]>, String> {
    // TODO(Morgan): Correct the Errors section in docs.
    let table = compute_table::<U>(reference, target, Penalties::default());
    let (aligned_x, aligned_y) = trace_back_recursive(&table, [reference, target]);
    Ok(serialize_edits(&unaligned_x_to_y(&aligned_x, &aligned_y)))
}

/// Decodes a reference string from a byte array.
///
/// # Arguments
///
/// * `reference`: The reference string.
/// * `encoding`: The byte array encoding the target string.
///
/// # Errors
///
/// * If the byte array cannot be deserialized into a vector of edits.
///
/// # Returns
///
/// The target string.
#[allow(dead_code, clippy::ptr_arg)]
pub fn decode_general(reference: &String, encoding: &[u8]) -> Result<String, String> {
    // TODO(Morgan): Correct the Errors section in docs.
    Ok(apply_edits(reference, &deserialize_edits(encoding)?))
}

/// Serializes a vector of edit operations into a byte array.
fn serialize_edits(edits: &[Edit]) -> Box<[u8]> {
    let bytes = edits.iter().flat_map(edit_to_bin).collect::<Vec<_>>();
    bytes.into_boxed_slice()
}

/// Deserializes a byte array into a vector of edit operations.
fn deserialize_edits(bytes: &[u8]) -> Result<Vec<Edit>, String> {
    let mut edits = Vec::new();
    let mut i = 0;
    let e_mask = 0b1100_0000;
    let v_mask = 0b0011_1111;
    while i < bytes.len() {
        let val = bytes[i] & e_mask;
        let edit = match val {
            0b1000_0000 => {
                let index = u16::from_be_bytes([bytes[i] & v_mask, bytes[i + 1]]);
                i += 2;
                Edit::Del(index as usize)
            }
            0b0100_0000 => {
                let index = u16::from_be_bytes([bytes[i] & v_mask, bytes[i + 1]]);
                let c = bytes[i + 2];
                i += 3;
                Edit::Ins(index as usize, c as char)
            }
            0b1100_0000 => {
                let index = u16::from_be_bytes([bytes[i] & v_mask, bytes[i + 1]]);
                let c = bytes[i + 2];
                i += 3;
                Edit::Sub(index as usize, c as char)
            }
            _ => return Err(format!("Invalid edit type: {val:b}.")),
        };
        edits.push(edit);
    }
    Ok(edits)
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
fn edit_to_bin(edit: &Edit) -> Vec<u8> {
    match edit {
        // First 2 bits for the type of edit, 14 bits for the index.
        Edit::Del(i) => {
            let mut bytes = (*i as u16).to_be_bytes().to_vec();
            bytes[0] |= 0b1000_0000;
            bytes
        }
        Edit::Ins(i, c) => {
            // First 2 bits for the type of edit, 14 bits for the index.
            let mut bytes = (*i as u16).to_be_bytes().to_vec();
            bytes[0] |= 0b0100_0000;
            // 8 bits for the character.
            bytes.push(*c as u8);
            bytes
        }
        Edit::Sub(i, c) => {
            // First 2 bits for the type of edit, 14 bits for the index.
            let mut bytes = (*i as u16).to_be_bytes().to_vec();
            bytes[0] |= 0b1100_0000;
            // 8 bits for the character.
            bytes.push(*c as u8);
            bytes
        }
    }
}
