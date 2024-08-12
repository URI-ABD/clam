//! Compression and Decompression with CLAM

mod codec_data;
mod compression;
mod decompression;
mod squishy_ball;

use distances::{number::Float, strings::needleman_wunsch, Number};

#[allow(clippy::module_name_repetitions)]
pub use codec_data::CodecData;
pub use compression::{Compressible, Encodable, ParCompressible};
pub use decompression::{Decodable, Decompressible, ParDecompressible};
pub use squishy_ball::SquishyBall;

use crate::{cluster::ParCluster, Cluster, FlatVec};

impl<I: Encodable, U: Number, M> Compressible<I, U> for FlatVec<I, U, M> {}
impl<I: Encodable + Send + Sync, U: Number, M: Send + Sync> ParCompressible<I, U> for FlatVec<I, U, M> {}

impl<I: Encodable + Decodable, U: Number, Co: Compressible<I, U>, Dec: Decompressible<I, U>, S: Cluster<I, U, Co>>
    super::cluster::Searchable<I, U, Dec> for SquishyBall<I, U, Co, Dec, S>
{
}

impl<
        I: Encodable + Decodable + Send + Sync,
        U: Number,
        Co: ParCompressible<I, U>,
        Dec: ParDecompressible<I, U>,
        S: ParCluster<I, U, Co>,
    > super::cluster::ParSearchable<I, U, Dec> for SquishyBall<I, U, Co, Dec, S>
{
}

/// Reads an encoded value from a byte array and increments the offset.
pub fn read_encoding(bytes: &[u8], offset: &mut usize) -> Box<[u8]> {
    let len = read_usize(bytes, offset);
    let encoding = bytes[*offset..*offset + len].to_vec();
    *offset += len;
    encoding.into_boxed_slice()
}

/// Reads a `usize` from a byte array and increments the offset.
pub fn read_usize(bytes: &[u8], offset: &mut usize) -> usize {
    let index_bytes: [u8; core::mem::size_of::<usize>()] = bytes[*offset..*offset + core::mem::size_of::<usize>()]
        .try_into()
        .unwrap_or_else(|e| unreachable!("Could not convert slice into array: {e:?}"));
    *offset += core::mem::size_of::<usize>();
    usize::from_le_bytes(index_bytes)
}

impl<F: Float> Encodable for Vec<F> {
    fn as_bytes(&self) -> Box<[u8]> {
        self.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    fn encode(&self, reference: &Self) -> Box<[u8]> {
        let diffs = reference.iter().zip(self.iter()).map(|(&a, &b)| a - b).collect();
        Self::as_bytes(&diffs)
    }
}

impl<F: Float> Decodable for Vec<F> {
    fn from_bytes(bytes: &[u8]) -> Self {
        bytes
            .chunks_exact(std::mem::size_of::<F>())
            .map(F::from_le_bytes)
            .collect()
    }

    fn decode(reference: &Self, bytes: &[u8]) -> Self {
        let diffs = Self::from_bytes(bytes);
        reference.iter().zip(diffs).map(|(&a, b)| a - b).collect()
    }
}

/// This uses the Needleman-Wunsch algorithm to encode strings.
impl Encodable for String {
    fn as_bytes(&self) -> Box<[u8]> {
        self.as_bytes().to_vec().into_boxed_slice()
    }

    fn encode(&self, reference: &Self) -> Box<[u8]> {
        let penalties = distances::strings::Penalties::default();
        let table = needleman_wunsch::compute_table::<u16>(reference, self, penalties);
        let (aligned_ref, aligned_tar) = needleman_wunsch::trace_back_recursive(&table, [reference, self]);
        let edits = needleman_wunsch::unaligned_x_to_y(&aligned_ref, &aligned_tar);
        serialize_edits(&edits)
    }
}

/// This uses the Needleman-Wunsch algorithm to decode strings.
impl Decodable for String {
    fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_utf8(bytes.to_vec()).unwrap_or_else(|e| unreachable!("Could not cast back to string: {e:?}"))
    }

    fn decode(reference: &Self, bytes: &[u8]) -> Self {
        let edits = deserialize_edits(bytes);
        needleman_wunsch::apply_edits(reference, &edits)
    }
}

/// Serializes a vector of edit operations into a byte array.
fn serialize_edits(edits: &[needleman_wunsch::Edit]) -> Box<[u8]> {
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
fn edit_to_bin(edit: &needleman_wunsch::Edit) -> Vec<u8> {
    let mask_idx = 0b00_111111;
    let mask_del = 0b10_000000;
    let mask_ins = 0b01_000000;
    let mask_sub = 0b11_000000;

    match edit {
        needleman_wunsch::Edit::Del(i) => {
            let mut bytes = (*i as u16).to_be_bytes().to_vec();
            // First 2 bits for the type of edit, 14 bits for the index.
            bytes[0] &= mask_idx;
            bytes[0] |= mask_del;
            bytes
        }
        needleman_wunsch::Edit::Ins(i, c) => {
            let mut bytes = (*i as u16).to_be_bytes().to_vec();
            // First 2 bits for the type of edit, 14 bits for the index.
            bytes[0] &= mask_idx;
            bytes[0] |= mask_ins;
            // 8 bits for the character.
            bytes.push(*c as u8);
            bytes
        }
        needleman_wunsch::Edit::Sub(i, c) => {
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
fn deserialize_edits(bytes: &[u8]) -> Vec<needleman_wunsch::Edit> {
    let mut edits = Vec::new();
    let mut offset = 0;
    let mask_idx = 0b00_111111;

    while offset < bytes.len() {
        let edit_bits = bytes[offset] & !mask_idx;
        let i = u16::from_be_bytes([bytes[offset] & mask_idx, bytes[offset + 1]]) as usize;
        let edit = match edit_bits {
            0b10_000000 => {
                offset += 2;
                needleman_wunsch::Edit::Del(i)
            }
            0b01_000000 => {
                let c = bytes[offset + 2] as char;
                offset += 3;
                needleman_wunsch::Edit::Ins(i, c)
            }
            0b11_000000 => {
                let c = bytes[offset + 2] as char;
                offset += 3;
                needleman_wunsch::Edit::Sub(i, c)
            }
            _ => unreachable!("Invalid edit type: {edit_bits:b}."),
        };
        edits.push(edit);
    }
    edits
}

#[cfg(test)]
pub mod tests {
    use crate::{adapter::BallAdapter, cakes::CodecData, Ball, Cluster, FlatVec, MetricSpace, Partition};

    use super::SquishyBall;

    use crate::cakes::tests::gen_random_data;

    #[test]
    fn ser_de() -> Result<(), String> {
        // The instances.
        type I = Vec<f64>;
        // The distance values.
        type U = f64;
        // The compressible dataset
        type Co = FlatVec<I, U, usize>;
        // The ball for the compressible dataset.
        type B = Ball<I, U, Co>;
        // The decompressible dataset
        type Dec = CodecData<I, U, usize>;
        // The squishy ball
        type Sb = SquishyBall<I, U, Co, Dec, B>;

        let seed = 42;
        let car = 1_000;
        let dim = 10;

        let data: Co = gen_random_data(car, dim, 10.0, seed)?;
        let metric = data.metric().clone();
        let metadata = data.metadata().to_vec();

        let criteria = |c: &B| c.cardinality() > 1;
        let ball = B::new_tree(&data, &criteria, Some(seed));
        let (_, co_data) = Sb::from_ball_tree(ball, data);
        let co_data = co_data.with_metadata(metadata.clone())?;

        let serialized = bincode::serialize(&co_data).unwrap();
        let deserialized = bincode::deserialize::<Dec>(&serialized)
            .unwrap()
            .post_deserialization(co_data.permutation.clone(), metadata)?
            .with_metric(metric.clone());

        assert_eq!(co_data.cardinality, deserialized.cardinality);
        assert_eq!(co_data.dimensionality_hint, deserialized.dimensionality_hint);
        assert_eq!(co_data.metadata, deserialized.metadata);
        assert_eq!(co_data.permutation, deserialized.permutation);
        assert_eq!(co_data.center_map, deserialized.center_map);
        assert_eq!(co_data.leaf_bytes, deserialized.leaf_bytes);
        assert_eq!(co_data.leaf_offsets, deserialized.leaf_offsets);
        assert_eq!(co_data.cumulative_cardinalities, deserialized.cumulative_cardinalities);

        Ok(())
    }
}
