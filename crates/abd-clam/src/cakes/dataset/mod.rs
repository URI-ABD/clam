//! Extension traits of `Dataset` for specific search applications.

mod codec_data;
mod compression;
mod decompression;
mod searchable;
mod shardable;

use distances::Number;

pub use codec_data::CodecData;
pub use compression::{Compressible, Encodable};
pub use decompression::{Decodable, Decompressible};
pub use searchable::{ParSearchable, Searchable};
pub use shardable::Shardable;

use crate::{cluster::ParCluster, Cluster, FlatVec};

impl<I: Encodable, U: Number, M> Compressible<I, U> for FlatVec<I, U, M> {}
impl<I, U: Number, C: Cluster<U>, M> Searchable<I, U, C> for FlatVec<I, U, M> {}

impl<I: Send + Sync, U: Number, C: ParCluster<U>, M: Send + Sync> ParSearchable<I, U, C> for FlatVec<I, U, M> {}

/// Reads an encoded value from a byte array and increments the offset.
pub(crate) fn read_encoding(bytes: &[u8], offset: &mut usize) -> Box<[u8]> {
    let len = read_usize(bytes, offset);
    let encoding = bytes[*offset..*offset + len].to_vec();
    *offset += len;
    encoding.into_boxed_slice()
}

/// Reads a `usize` from a byte array and increments the offset.
pub(crate) fn read_usize(bytes: &[u8], offset: &mut usize) -> usize {
    let index_bytes: [u8; std::mem::size_of::<usize>()] = bytes[*offset..*offset + std::mem::size_of::<usize>()]
        .try_into()
        .unwrap_or_else(|e| unreachable!("Could not convert slice into array: {e:?}"));
    *offset += std::mem::size_of::<usize>();
    usize::from_le_bytes(index_bytes)
}
