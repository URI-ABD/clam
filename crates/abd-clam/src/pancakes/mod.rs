//! Compression and Decompression with CLAM

mod cluster;
mod dataset;
mod sequence;

pub use cluster::{SquishCosts, SquishyBall};
pub use dataset::{CodecData, Compressible, Decodable, Decompressible, Encodable, ParCompressible, ParDecompressible};
