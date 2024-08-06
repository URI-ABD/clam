//! Compression and Decompression with CLAM

mod codec_data;
mod compression;
mod decompression;
mod squishy_ball;

use distances::Number;

#[allow(clippy::module_name_repetitions)]
pub use codec_data::CodecData;
pub use compression::{Compressible, Encodable};
pub use decompression::{Decodable, Decompressible};
pub use squishy_ball::SquishyBall;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, FlatVec};

impl<I: Encodable, U: Number, M> Compressible<I, U> for FlatVec<I, U, M> {}

impl<I: Decodable, U: Number, D: Decompressible<I, U>, S: Cluster<I, U, D>> super::cluster::Searchable<I, U, D>
    for SquishyBall<I, U, D, S>
{
}
impl<I: Decodable + Send + Sync, U: Number, D: Decompressible<I, U> + ParDataset<I, U>, S: ParCluster<I, U, D>>
    super::cluster::ParSearchable<I, U, D> for SquishyBall<I, U, D, S>
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
    let index_bytes: [u8; std::mem::size_of::<usize>()] = bytes[*offset..*offset + std::mem::size_of::<usize>()]
        .try_into()
        .unwrap_or_else(|e| unreachable!("Could not convert slice into array: {e:?}"));
    *offset += std::mem::size_of::<usize>();
    usize::from_le_bytes(index_bytes)
}
