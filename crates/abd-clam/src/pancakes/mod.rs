//! Compression and Decompression with CLAM

mod cluster;
mod dataset;

pub use cluster::{SquishCosts, SquishyBall};
pub use dataset::{
    CodecData, Compressible, Decoder, Decompressible, Encoder, ParCompressible, ParDecoder, ParDecompressible,
    ParEncoder,
};
